import io
import os
import onnx
from onnx import numpy_helper
import torch

from distutils.version import LooseVersion
from inspect import signature

import onnxruntime.capi.postprocess as postprocess
import onnxruntime as ort
from . import ORTTrainerOptions
from . import optim
from .model_desc_validation import _ORTTrainerModelDesc


class TrainStepInfo(object):
    r"""Private class used to store runtime information from current train step.

    After every train step, :py:meth:`ORTTrainer.train_step` updates the internal instance of
    :py:class:`.TrainStepInfo` residing on :py:class:`.ORTTrainer` with relevant information
    from the forward pass.

    This class shouldn't be accessed directly by the user, unless they really know what they are doing.
    Instead, :py:class:`.ORTTrainer` passes it to relevant class methods automatically,
    such as :py:method:`._LRScheduler.get_lr` or :py:class:`.LossScaler.update`.

    Args:
        all_finite (bool): flag that indicates whether all gradients are still finite after last step
        step (int): indicates current step
        optimizer_config (optim._OptimizerConfig): reference to optimizer config

    Example:

        .. code-block:: python

            info = TrainStepInfo(all_finite=True, step=0, optimizer_config=optim.SGDConfig(lr=0.01))
            if info.all_finite:
                print(f'Yay, all gradients are finite at {step} step!')

    """

    def __init__(self, all_finite=None, step=None, optimizer_config=None):
        assert all_finite is None or isinstance(all_finite, bool),\
            "all_finite must be either None or a bool"
        assert step is None or (isinstance(step, int) and step >= 0),\
            "step must be either None or a positive int"
        assert optimizer_config is None or isinstance(optimizer_config, optim._OptimizerConfig),\
            "optimizer_config must be either None or optim._OptimizerConfig"

        self.all_finite = all_finite
        self.step = step
        self.optimizer_config = optimizer_config


class ORTTrainer(object):
    r"""Pytorch frontend for ONNX Runtime training

    Entry point that exposes the C++ backend of ORT as a Pytorch frontend.

    Args:
        model (torch.nn.Module or onnx.ModelProto): either a PyTorch or ONNX model.
            When a PyTorch model and :py:attr:`loss_fn` are specified, :py:attr:`model` and :py:obj:`loss_fn` are combined.
            When a ONNX model is provided, the loss is identified by the flag :py:obj:`is_loss=True` in one of the :py:attr:`.model_desc.outputs` entries.
        model_desc (dict): model input and output description.
            This is used to identify inputs and outputs and their shapes, so that ORT can generate back propagation graph, plan memory allocation for
            training, and perform optimizations.
            :py:attr:`model_desc` must be consistent with the training :py:attr:`model` and have the following (:py:obj:`dict`) schema
            :py:obj:`{ 'inputs': [tuple(name, shape)], 'outputs': [tuple(name, shape, is_loss)]}`.
            :py:attr:`name` is a string representing the name of input or output of the model.
            For :py:obj:`model_desc['inputs']` entries, :py:attr:`name` must match input names of the original PyTorch model's :py:meth:`torch.nn.Module.forward` method.
            For ONNX models, both name and order of input names must match.
            For :py:obj:`model_desc['outputs']` entries, the order must match the original PyTorch's output as returned by :py:meth:`torch.nn.Module.forward` method.
            For ONNX models, both name and order of output names must match.
            :py:attr:`shape` is a list of string or integers that describes the shape of the input/output.
            Each dimension size can be either a string or an int. String means the dimension size is dynamic, while integers mean static dimensions.
            An empty list implies a scalar.
            Lastly, :py:attr:`is_loss` is a boolean (default is False) that flags if this output is considered a loss.
            ORT backend needs to know which output is loss in order to generate back propagation graph.
            Loss output must be specified when either :py:attr:`loss_fn` is specified or when loss is embedded in the model.
            Note that only one loss output is supported per model.
        optimizer_config (optim._OptimizerConfig): optimizer config.
            One of :py:class:`.optim.AdamConfig`, :py:class:`.optim.LambConfig` or :py:class:`.optim.SGDConfig`.
        loss_fn (callable, default is None): a PyTorch loss function.
            It takes two inputs [prediction, label] and outputs a scalar loss tensor.
            If provided, :py:attr:`loss_fn` is combined with the PyTorch :py:attr:`model` to form a combined PyTorch model.
            Inputs to the combined PyTorch model are concatenation of the :py:attr:`model`'s input and :py:attr:`loss_fn`'s label input.
            Outputs of the combined PyTorch model are concatenation of :py:attr:`loss_fn`'s loss output and :py:attr:`model`'s outputs.
        options (ORTTrainerOptions, default is None): options for additional features.

    Example:

        .. code-block:: python

            model = ...
            loss_fn = ...
            model_desc = {
                "inputs": [
                    ("input_ids", ["batch", "max_seq_len_in_batch"]),
                    ("attention_mask", ["batch", "max_seq_len_in_batch"]),
                    ("token_type_ids", ["batch", "max_seq_len_in_batch"]),
                    ("masked_lm_labels", ["batch", "max_seq_len_in_batch"]),
                    ("next_sentence_label", ["batch", 1])
                ],
                "outputs": [
                    ("loss", [], True),
                ],
            }
            optim_config = optim.LambConfig(param_groups = [ { 'params' : ['model_param0'], 'alpha' : 0.8, 'beta' : 0.7},
                                                             { 'params' : ['model_param1' , 'model_param_2'], 'alpha' : 0.0}
                                                           ],
                                            alpha=0.9, beta=0.999)
            ort_trainer = ORTTrainer(model, model_desc, optim_config, loss_fn)
    """

    def __init__(self, model, model_desc, optim_config, loss_fn=None, options=None):
        # Basic validation
        assert model is not None, "'model' is required and must be either a 'torch.nn.Module' or ONNX model"
        assert isinstance(model_desc, dict), "'model_desc' must be a 'dict'"
        assert isinstance(optim_config, optim._OptimizerConfig),\
            "'optim_config' is required and must be any of 'AdamConfig', 'LambConfig' or 'SGDConfig'"
        assert loss_fn is None or (callable(loss_fn) and len(signature(loss_fn).parameters) == 2),\
            "'loss_fn' must be either 'None' or a callable with two parameters"
        assert options is None or isinstance(options, ORTTrainerOptions),\
            "'loss_fn' must be either 'None' or 'ORTTrainerOptions'"

        #            Model + Loss validation
        #           Supported combinarios are
        #    ----------------------------------------
        #   |   | Model            | Loss            |
        #    ----------------------------------------
        #   | 1 | torch.nn.Module  | None            |
        #   | 2 | torch.nn.Module  | torch.nn.Module |
        #   | 3 | ONNX             | None            |
        #    ----------------------------------------
        self._torch_model = None
        self._onnx_model = None
        if isinstance(model, torch.nn.Module):
            assert loss_fn is None or isinstance(model, torch.nn.Module),\
                "'loss_fn' must be either 'None' or 'torch.nn.Module'"
            self._torch_model = model
            self.loss_fn = loss_fn
        elif isinstance(model, onnx.ModelProto):
            assert loss_fn is None, "'loss_fn' must not be specified when 'model' is an ONNX model"
            self._onnx_model = model
            self.loss_fn = None
        else:
            raise ValueError("'model' must be either 'torch.nn.Module' or 'onnx.ModelProto'")

        self.model_desc = _ORTTrainerModelDesc(model_desc)
        self.optim_config = optim_config

        if options:
            self.options = ORTTrainerOptions(options)
        else:
            self.options = ORTTrainerOptions()

    def eval_step(self, *input, **kwargs):
        r"""Evaluation step method

        Args:
            *input: Arbitrary arguments that are used as model input (data only)
            **kwargs: Arbitrary keyword arguments that are used as model input (data only)

        Returns:
            ordered :py:obj:`list` with model outputs as described by :py:attr:`.ORTTrainer.model_desc`
        """
        pass

    def save_as_onnx(self, path):
        r"""Persists ONNX model into :py:attr:`path`

        The model will be saved as a Google Protocol Buffers (aka protobuf) file as per ONNX standard.
        The graph includes full information, including inference and training metadata.

        Args:
            path (str): Full path, including filename, to save the ONNX model in the filesystem

        Raises:
            RuntimeWarning: raised when neither `train_step` or `eval_step` was called at least once
            ValueError: raised when `path` is not valid path
        """
        if not self._training_session:
            raise RuntimeWarning("Training session is not initialized yet. "
                                 "'train_step' or 'eval_step' methods must be executed at least once before calling 'save_as_onnx()'.")
        state_tensors = self._training_session.get_state()
        self._update_onnx_model_initializers(state_tensors)

        assert isinstance(path, str), "'path' must be a valid path string"
        dir_name = os.path.dirname(path)
        file_name = os.path.basename(path)
        if not dir_name or not os.path.exists(dir_name) or not file_name:
            raise ValueError("'path' must be a valid path string")

        with open(path, "wb") as f:
            f.write(self._onnx_model.SerializeToString())

    def train_step(self, *input, **kwargs):
        r"""Train step method

        After forward pass, an ordered list with all outputs described at :py:attr:`ORTTrainer.model_desc` is returned.
        Additional information relevant to the train step is maintend by :py:attr:`ORTTrainer._train_step_info`.
        See :py:class:`.TrainStepInfo` for details.

        Args:
            *input: Arbitrary arguments that are used as model input (data only)
            **kwargs: Arbitrary keyword arguments that are used as model input (data only)

        Returns:
            ordered :py:obj:`list` with model outputs as described by :py:attr:`ORTTrainer.model_desc`
        """

        # Export model to ONNX
        if self._onnx_model is None:
            sample_input = self._prepare_input_and_fetches(
                self.model_desc.inputs, None, None, *input, **kwargs)
            self._init_onnx_model(sample_input)

    def _combine_torch_model_with_loss(self):
        # Don't need to wrap model when loss_fn is not set
        if not self.loss_fn:
            return self._torch_model

        # Validate loss_fn
        sig_loss = signature(self.loss_fn)
        if len(sig_loss.parameters) != 2:
            raise RuntimeError(
                "loss function should take two arguments - predict and label.")

        # Basic input names from model
        input_names = [input[0] for input in self.model_desc.inputs]
        sig = signature(self._torch_model.forward)
        ordered_input_list = list(sig.parameters.keys())

        # Label from loss_fn goes after model input
        ordered_input_list = [*ordered_input_list,
                              list(sig_loss.parameters.keys())[1]]

        # Check whether input names from model match inputs from ModelDescription
        match = True
        for ordered_list_key, input_name in zip(ordered_input_list, input_names):
            if ordered_list_key != input_name:
                match = False
                break

        # Input can be a list or dict
        is_list_input = (match
                         or len(input_names) >= len(ordered_input_list)
                         or not all(x in ordered_list_kes for x in input_names))

        class CombineTorchModelLossFn(torch.nn.Module):
            def __init__(self, model, loss_fn, input_names):
                super(CombineTorchModelLossFn, self).__init__()
                self.model = model
                self.loss_fn = loss_fn
                self.input_names = input_names

            def forward(self, *inputs):
                # *inputs is given by torch trace. It is in the order of input_names.
                # model_ takes input in a order (which can be obtained via inspect.signature(model.forward)) different than input_names.
                if is_list_input:
                    input, label = inputs[:-1], inputs[-1]
                    preds = self.model(*input)
                    # TODO: order this according to self.model_desc.outputs?
                    return self.loss_fn(preds, label), preds
                else:
                    sig = signature(self.model.forward)
                    ordered_input_list = list(sig.parameters.keys())

                    input_dict = {}
                    for key in sig.parameters.keys():
                        if key in self.input_names:
                            input_dict[key] = inputs[self.input_names.index(key)]

                    model_out = self.model(**input_dict)
                    if self.loss_fn is None:
                        return model_out

                    label = inputs[-1]
                    preds = model_out
                    return self.loss_fn(preds, label), preds

        return CombineTorchModelLossFn(self._torch_model, self.loss_fn, input_names)

    def _convert_torch_model_loss_fn_to_onnx(self, device, inputs):
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]
        if isinstance(inputs, dict):
            sample_inputs = [inputs[k.name_].to(
                device=device) for k in self.model_desc.inputs]
        elif isinstance(inputs, (list, tuple)):
            sample_inputs = [input.to(device=device) for i, input in enumerate(
                inputs) if i < len(self.model_desc.inputs)]
        else:
            raise RuntimeError(
                "Unexpected input type. Only torch.Tensor, or dict/list/tuple of torch.Tensor is supported.")

        # pytorch onnx exporter/trace does not try to match argument names.
        # e.g. for models with optional inputs, it requires all inputs be present.
        # this is a problem because the model graph depends on inputs provided.
        model = self._combine_torch_model_with_loss()

        # Do an inference to grab output types
        model.eval()
        with torch.no_grad():
            sample_outputs = model(*sample_inputs)
        model.train()
        if isinstance(sample_outputs, torch.Tensor):
            sample_outputs = [sample_outputs]

        # Append 'dtypes' for model description inputs/outputs
        for i, sample_input in enumerate(sample_inputs):
            if i < len(self.model_desc.inputs):
                self.model_desc.add_type_to_input_description(
                    i, sample_input.dtype)
        for i, sample_output in enumerate(sample_outputs):
            if i < len(self.model_desc.outputs):
                self.model_desc.add_type_to_output_description(
                    i, sample_output.dtype)

        # Export the model to ONNX
        f = io.BytesIO()
        torch.onnx._export(model, tuple(sample_inputs), f,
                           input_names=[input[0] for input in self.model_desc.inputs],
                           output_names=[output[0] for output in self.model_desc.outputs],
                           opset_version=self.options._internal_use.onnx_opset_version,
                           _retain_param_name=True,
                           example_outputs=tuple(sample_outputs),
                           do_constant_folding=False,
                           training=torch.onnx.TrainingMode.TRAINING)
        onnx_model = onnx.load_model_from_string(f.getvalue())

        # Remove 'model.' prefix introduced by model wrapper for initializers.
        replace_name_dict = {}
        for n in onnx_model.graph.initializer:
            if n.name.startswith('model.'):
                replace_name_dict[n.name] = n.name[len('model.'):]
                n.name = replace_name_dict[n.name]
        for n in onnx_model.graph.node:
            for i, name in enumerate(n.input):
                if name in replace_name_dict:
                    n.input[i] = replace_name_dict[name]

        # onnx model initializer may contain non-trainable registered buffers that are not part
        # of pytorch model named parameteres.
        named_parameters = model.model.named_parameters() if hasattr(model, 'model') else model.named_parameters()
        assert set([n for n, t in named_parameters]).issubset(
            set([n.name for n in onnx_model.graph.initializer])), \
            "Initializer names do not match between PyTorch model and ONNX model, " \
            "please report a bug to ONNX Runtime."

        return onnx_model

    def _init_onnx_model(self, inputs):
        if self._onnx_model is not None:
            return

        if self._torch_model is not None:
            # PyTorch model is moved to cpu to save GPU memory
            self._torch_model.cpu()

            # PyTorch buffers (created using 'register_buffer') shouldn't be trained
            torch_buffers = list(dict(self._torch_model.named_buffers()).keys())
            self.options.utils.frozen_weights.extend(torch_buffers)

            # Export to ONNX
            self._onnx_model = self._convert_torch_model_loss_fn_to_onnx(torch.device('cpu'), inputs)

        self._init_session()

    def _init_session(self):
        if self._onnx_model is None:
            return

        if self.options._internal_use.enable_internal_postprocess:
            self._onnx_model = postprocess.run_postprocess(self._onnx_model)
        if self.options._internal_use.extra_postprocess:
            self.options._internal_use.extra_postprocess(self._onnx_model)
        
        # create the training session
        return
    
    def _prepare_input_and_fetches(self, inputs_desc, lr, loss_scale, *args, **kwargs):
        # Normalize input to tuple of samples
        if type(args) == tuple and len(args) == 1 and type(args[0]) == list:
            input = tuple(args[0])
        else:
            input = args

        # Append input from kwargs
        for input_desc in inputs_desc:
            if input_desc[0] in kwargs:
                input = input + (kwargs[input_desc[0]],)

        return input

    def _update_onnx_model_initializers(self, state_tensors):
        r""" Updates ONNX graph initializers with state_tensors's values

        Usually called to save or load an ONNX model.

        The tensors names of state_tensors are compared to all ONNX initializer tensors
        and when the name matches, the ONNX graph is updated with the new value.
        """
        assert isinstance(state_tensors, dict), "state_tensors must be a dict"

        new_weights = []
        replace_indices = []
        for i, w in enumerate(self._onnx_model.graph.initializer):
            if w.name in state_tensors:
                new_weights.append(numpy_helper.from_array(state_tensors[w.name], w.name))
                replace_indices.append(i)
        replace_indices.sort(reverse=True)
        for w_i in replace_indices:
            del self._onnx_model.graph.initializer[w_i]
        self._onnx_model.graph.initializer.extend(new_weights)

    def _create_ort_training_session(self, use_deterministic_compute=False):
        # Validating frozen_weights names
        unused_frozen_weights = [n for n in self.options.utils.frozen_weights if n not in [i.name for i in self._onnx_model.graph.initializer]]
        if unused_frozen_weights:
            raise RuntimeError("{} params from 'frozen_weights' not found in the ONNX model.".format(unused_frozen_weights))

        # Get loss name from model description
        loss_name = [item.name for item in self.model_desc.outputs if len(item) == 4 and item[2]]
        assert len(loss_name) == 1, f"Only one loss output is supported ({len(loss_name)} were specified)"
        loss_name = loss_name[0]

        # Parse optimizer parameters
        optimizer_attributes_map = {}
        optimizer_int_attributes_map = {}
        trainable_params = set()
        for initializer in self._onnx_model.graph.initializer:
            if initializer.name in self.options.utils.frozen_weights:
                continue  # only trainable parameters are passed to the backend
            trainable_params.add(initializer.name)
            optimizer_attributes_map[initializer.name] = {}
            optimizer_int_attributes_map[initializer.name] = {}
            for param_group in self.optim_config.params:
                if initializer.name not in param_group['params']:
                    continue  # keep looking for a matching param_group
                optimizer_attributes_map[initializer.name] = {}
                optimizer_int_attributes_map[initializer.name] = {}
                for k, v in param_group.items():
                    if k == 'params':
                        continue  # 'params' is not a hyper parameter, skip it
                    if isinstance(v, float):
                        optimizer_attributes_map[initializer.name][k] = v
                    elif isinstance(v, int):
                        optimizer_int_attributes_map[initializer.name][k] = v
                    else:
                        raise ValueError("Optimizer attributes must be either float or int.")

        # TrainingParameters
        ort_parameters = ort.TrainingParameters()
        ort_parameters.loss_output_name = loss_name
        ort_parameters.use_mixed_precision = self.options.mixed_precision.enabled
        ort_parameters.world_rank = self.options.distributed.world_rank
        ort_parameters.world_size = self.options.distributed.world_size
        ort_parameters.gradient_accumulation_steps = self.options.batch.gradient_accumulation_steps
        ort_parameters.allreduce_post_accumulation = self.options.distributed.allreduce_post_accumulation
        ort_parameters.deepspeed_zero_stage = self.options.distributed.deepspeed_zero_stage
        ort_parameters.enable_grad_norm_clip = self.options.utils.grad_norm_clip
        ort_parameters.set_gradients_as_graph_outputs = False
        ort_parameters.training_optimizer_name = self.optim_config.name
        
        print(type(ort_parameters.lr_params_feed_name), type(self.optim_config.lr))

        ort_parameters.lr_params_feed_name = str(self.optim_config.lr)
        #ort_parameters.trainable_params = trainable_params
        ort_parameters.weights_to_train = trainable_params
        ort_parameters.optimizer_attributes_map = optimizer_attributes_map
        ort_parameters.optimizer_int_attributes_map = optimizer_int_attributes_map

        # SessionOptions
        session_options = ort.SessionOptions()
        session_options.use_deterministic_compute = use_deterministic_compute

        # TrainingSession
        self._training_session = ort.TrainingSession(self._onnx_model.SerializeToString(), ort_parameters, session_options)

        # I/O bindings
        self._train_io_binding = self._training_session.io_binding()
        self._eval_io_binding = self._training_session.io_binding()
