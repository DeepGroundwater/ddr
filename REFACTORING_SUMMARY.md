# Muskingum-Cunge Refactoring Summary

This document summarizes the refactoring of the Muskingum-Cunge routing implementation from `dmc.py` into two new modular files.

## Files Created

### 1. `src/ddr/routing/mmc.py` - Core Mathematical Implementation

**Purpose**: Contains the pure mathematical implementation of the Muskingum-Cunge routing algorithm.

**Key Features**:
- `MuskingunCunge` class that manages all hydrofabric data, parameters, and routing calculations
- Self-contained implementation that handles data setup and management internally
- Clean separation of mathematical logic from PyTorch neural network concerns
- Comprehensive input validation and parameter management

**Main Methods**:
- `setup_inputs()`: Configures all inputs (hydrofabric, streamflow, spatial parameters)
- `forward()`: Performs complete routing calculation through time series
- `route_timestep()`: Routes flow for a single timestep
- `calculate_muskingum_coefficients()`: Computes routing coefficients
- Utility methods for sparse matrix operations

### 2. `src/ddr/routing/torch_mc.py` - PyTorch Neural Network Module

**Purpose**: Provides a PyTorch `nn.Module` wrapper around the core implementation for training and inference.

**Key Features**:
- `TorchMC` class that inherits from `torch.nn.Module`
- GPU/CPU device management with proper tensor movement
- Full compatibility with PyTorch training pipelines
- Backward compatibility with the original `dmc` interface
- Automatic differentiation support for training

**Main Methods**:
- `forward()`: PyTorch forward pass using the core implementation
- `to()`, `cuda()`, `cpu()`: Device management methods
- `state_dict()`, `load_state_dict()`: Model saving/loading
- Compatibility methods that delegate to the core implementation

## Key Improvements

### 1. **Separation of Concerns**
- Mathematical routing logic separated from PyTorch-specific functionality
- Core implementation is framework-agnostic and can be reused
- Neural network wrapper provides PyTorch integration

### 2. **Data Management**
- `MuskingunCunge` class now manages all input data internally
- Eliminates need to pass hydrofabric data through multiple method calls
- Cleaner, more maintainable interface

### 3. **Backward Compatibility**
- `TorchMC` is aliased as `dmc` for drop-in replacement capability
- Existing training and evaluation scripts require no changes
- All original method signatures preserved through compatibility methods

### 4. **Device Flexibility**
- Proper PyTorch device management with `to()`, `cuda()`, `cpu()` methods
- Tensors automatically moved to correct devices
- GPU/CPU training fully supported

### 5. **Code Organization**
- Better structured codebase with clear responsibilities
- Easier to test and maintain individual components
- More modular architecture supports future extensions

## Usage Examples

### Using the Core Implementation
```python
from ddr.routing.mmc import MuskingunCunge

# Create routing engine
mc = MuskingunCunge(cfg, device='cuda')

# Setup all inputs at once
mc.setup_inputs(hydrofabric, streamflow, spatial_parameters)

# Run routing
output = mc.forward()
```

### Using the PyTorch Module
```python
from ddr.routing.torch_mc import TorchMC

# Create PyTorch module
model = TorchMC(cfg, device='cuda')

# Use like any PyTorch module
kwargs = {
    'hydrofabric': hydrofabric,
    'streamflow': streamflow,
    'spatial_parameters': spatial_parameters
}
output = model(**kwargs)

# Training compatibility
loss = criterion(output['runoff'], targets)
loss.backward()
```

### Drop-in Replacement
```python
# Original code - no changes needed
from ddr.routing.dmc import dmc

routing_model = dmc(cfg=cfg, device=cfg.device)
output = routing_model(**dmc_kwargs)
```

## Migration Guide

### For Existing Code
- **No changes required** - the `TorchMC` class provides full backward compatibility
- Import statements continue to work: `from ddr.routing.dmc import dmc`
- All existing method calls and interfaces preserved

### For New Development
- Use `MuskingunCunge` directly for pure routing calculations
- Use `TorchMC` for PyTorch training and inference
- Leverage the improved data management for cleaner code

## Testing

A test script `test_refactor.py` is provided to verify:
- Core `MuskingunCunge` functionality
- `TorchMC` PyTorch module compatibility
- Device movement capabilities
- Backward compatibility

## Benefits

1. **Maintainability**: Clear separation of concerns makes code easier to understand and maintain
2. **Testability**: Individual components can be tested in isolation
3. **Reusability**: Core routing logic can be used in different contexts
4. **Extensibility**: Modular design supports future enhancements
5. **Performance**: More efficient data management reduces overhead
6. **Compatibility**: Zero-impact migration for existing code

The refactoring successfully achieves the goal of creating a more modular, maintainable, and flexible Muskingum-Cunge routing implementation while preserving full backward compatibility.
