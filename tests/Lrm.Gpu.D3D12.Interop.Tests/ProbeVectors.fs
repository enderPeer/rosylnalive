/// Known answers recorded by the planning probe (LRM_DATA/spikes-staging/probe/d3d12probe.fsx), which called these
/// slots by hand on the RTX 4080 and ran a ResourceDescriptorHeap dispatch with them. The generated slots, IIDs and
/// layouts must reproduce them exactly (dev plan section 8, gate 3: "D3D12 slots match the probe").
module Lrm.Gpu.D3D12.Interop.Tests.ProbeVectors

open System
open Lrm.Gpu.D3D12.Interop

/// (interface, method, generated slot, probe slot).
let slots =
    [
        "IUnknown", "QueryInterface", Slots.IUnknown.QueryInterface, 0
        "IUnknown", "Release", Slots.IUnknown.Release, 2
        "IDXGIFactory6", "EnumAdapterByGpuPreference", Slots.IDXGIFactory6.EnumAdapterByGpuPreference, 29
        "IDXGIAdapter1", "GetDesc1", Slots.IDXGIAdapter1.GetDesc1, 10
        "ID3D12Device", "CreateCommandQueue", Slots.ID3D12Device.CreateCommandQueue, 8
        "ID3D12Device", "CreateCommandAllocator", Slots.ID3D12Device.CreateCommandAllocator, 9
        "ID3D12Device", "CreateComputePipelineState", Slots.ID3D12Device.CreateComputePipelineState, 11
        "ID3D12Device", "CreateCommandList", Slots.ID3D12Device.CreateCommandList, 12
        "ID3D12Device", "CheckFeatureSupport", Slots.ID3D12Device.CheckFeatureSupport, 13
        "ID3D12Device", "CreateDescriptorHeap", Slots.ID3D12Device.CreateDescriptorHeap, 14
        "ID3D12Device", "GetDescriptorHandleIncrementSize", Slots.ID3D12Device.GetDescriptorHandleIncrementSize, 15
        "ID3D12Device", "CreateRootSignature", Slots.ID3D12Device.CreateRootSignature, 16
        "ID3D12Device", "CreateUnorderedAccessView", Slots.ID3D12Device.CreateUnorderedAccessView, 19
        "ID3D12Device", "CreateCommittedResource", Slots.ID3D12Device.CreateCommittedResource, 27
        "ID3D12Device", "CreateFence", Slots.ID3D12Device.CreateFence, 36
        "ID3D12Device", "GetDeviceRemovedReason", Slots.ID3D12Device.GetDeviceRemovedReason, 37
        "ID3D12DeviceConfiguration", "GetDesc", Slots.ID3D12DeviceConfiguration.GetDesc, 3
        "ID3D12DescriptorHeap",
        "GetCPUDescriptorHandleForHeapStart",
        Slots.ID3D12DescriptorHeap.GetCPUDescriptorHandleForHeapStart,
        9
        "ID3D12Resource", "Map", Slots.ID3D12Resource.Map, 8
        "ID3D12Resource", "Unmap", Slots.ID3D12Resource.Unmap, 9
        "ID3D12CommandQueue", "ExecuteCommandLists", Slots.ID3D12CommandQueue.ExecuteCommandLists, 10
        "ID3D12CommandQueue", "Signal", Slots.ID3D12CommandQueue.Signal, 14
        "ID3D12Fence", "SetEventOnCompletion", Slots.ID3D12Fence.SetEventOnCompletion, 9
        "ID3D12GraphicsCommandList", "Close", Slots.ID3D12GraphicsCommandList.Close, 9
        "ID3D12GraphicsCommandList", "Dispatch", Slots.ID3D12GraphicsCommandList.Dispatch, 14
        "ID3D12GraphicsCommandList", "CopyBufferRegion", Slots.ID3D12GraphicsCommandList.CopyBufferRegion, 15
        "ID3D12GraphicsCommandList", "ResourceBarrier", Slots.ID3D12GraphicsCommandList.ResourceBarrier, 26
        "ID3D12GraphicsCommandList", "SetDescriptorHeaps", Slots.ID3D12GraphicsCommandList.SetDescriptorHeaps, 28
        "ID3D12GraphicsCommandList",
        "SetComputeRootSignature",
        Slots.ID3D12GraphicsCommandList.SetComputeRootSignature,
        29
        "ID3D12GraphicsCommandList",
        "SetComputeRoot32BitConstants",
        Slots.ID3D12GraphicsCommandList.SetComputeRoot32BitConstants,
        35
    ]

/// (interface, generated IID, probe IID).
let iids =
    [
        "ID3D12Device", Iids.ID3D12Device, Guid "189819f1-1db6-4b57-be54-1821339b85f7"
        "ID3D12CommandQueue", Iids.ID3D12CommandQueue, Guid "0ec870a6-5d7e-4c22-8cfc-5baae07616ed"
        "ID3D12CommandAllocator", Iids.ID3D12CommandAllocator, Guid "6102dee4-af59-4b09-b999-b44d73f09b24"
        "ID3D12GraphicsCommandList", Iids.ID3D12GraphicsCommandList, Guid "5b160d0f-ac1b-4185-8ba8-b3ae42a5a455"
        "ID3D12Fence", Iids.ID3D12Fence, Guid "0a753dcf-c4d8-4b91-adf6-be5a60d95a76"
        "ID3D12Resource", Iids.ID3D12Resource, Guid "696442be-a72e-4059-bc79-5b5c98040fad"
        "ID3D12DescriptorHeap", Iids.ID3D12DescriptorHeap, Guid "8efb471d-616c-4f49-90f7-127bb763fa51"
        "ID3D12RootSignature", Iids.ID3D12RootSignature, Guid "c54a6b66-72df-4ee8-8be5-a946a1429214"
        "ID3D12PipelineState", Iids.ID3D12PipelineState, Guid "765a30f3-f624-4c6f-a828-ace948622445"
        "ID3D12DeviceConfiguration", Iids.ID3D12DeviceConfiguration, Guid "78dbf87b-f766-422b-a61c-c8c446bdb9ad"
        "IDXGIFactory6", Iids.IDXGIFactory6, Guid "c1b6694f-ff09-44a9-b03c-77900a0a1d17"
        "IDXGIAdapter1", Iids.IDXGIAdapter1, Guid "29038f61-3839-4626-91fd-086879011a05"
        "IUnknown", Iids.IUnknown, Guid "00000000-0000-0000-c000-000000000046"
    ]

/// (struct, generated type, probe size): the buffer sizes the probe allocated and filled by hand.
let sizes: (string * Type * int) list =
    [
        "D3D12_COMMAND_QUEUE_DESC", typeof<D3D12_COMMAND_QUEUE_DESC>, 16
        "D3D12_HEAP_PROPERTIES", typeof<D3D12_HEAP_PROPERTIES>, 20
        "D3D12_RESOURCE_DESC", typeof<D3D12_RESOURCE_DESC>, 56
        "D3D12_DESCRIPTOR_HEAP_DESC", typeof<D3D12_DESCRIPTOR_HEAP_DESC>, 16
        "D3D12_UNORDERED_ACCESS_VIEW_DESC", typeof<D3D12_UNORDERED_ACCESS_VIEW_DESC>, 40
        "D3D12_RESOURCE_BARRIER", typeof<D3D12_RESOURCE_BARRIER>, 32
        "D3D12_COMPUTE_PIPELINE_STATE_DESC", typeof<D3D12_COMPUTE_PIPELINE_STATE_DESC>, 56
        "D3D12_FEATURE_DATA_D3D12_OPTIONS", typeof<D3D12_FEATURE_DATA_D3D12_OPTIONS>, 60
        "D3D12_FEATURE_DATA_D3D12_OPTIONS1", typeof<D3D12_FEATURE_DATA_D3D12_OPTIONS1>, 24
        "D3D12_FEATURE_DATA_D3D12_OPTIONS4", typeof<D3D12_FEATURE_DATA_D3D12_OPTIONS4>, 12
        "D3D12_FEATURE_DATA_D3D12_OPTIONS9", typeof<D3D12_FEATURE_DATA_D3D12_OPTIONS9>, 24
        "D3D12_FEATURE_DATA_D3D12_OPTIONS11", typeof<D3D12_FEATURE_DATA_D3D12_OPTIONS11>, 4
        "D3D12_FEATURE_DATA_SHADER_MODEL", typeof<D3D12_FEATURE_DATA_SHADER_MODEL>, 4
        "D3D12_DEVICE_CONFIGURATION_DESC", typeof<D3D12_DEVICE_CONFIGURATION_DESC>, 16
    ]

/// (struct, field, generated type, probe offset): the offsets the probe wrote and read.
let offsets: (Type * string * int) list =
    [
        typeof<D3D12_RESOURCE_DESC>, "Width", 16
        typeof<D3D12_RESOURCE_DESC>, "Height", 24
        typeof<D3D12_RESOURCE_DESC>, "DepthOrArraySize", 28
        typeof<D3D12_RESOURCE_DESC>, "MipLevels", 30
        typeof<D3D12_RESOURCE_DESC>, "SampleDesc", 36
        typeof<D3D12_RESOURCE_DESC>, "Layout", 44
        typeof<D3D12_RESOURCE_DESC>, "Flags", 48
        typeof<D3D12_UNORDERED_ACCESS_VIEW_DESC>, "ViewDimension", 4
        typeof<D3D12_UNORDERED_ACCESS_VIEW_DESC>, "Buffer", 8
        typeof<D3D12_BUFFER_UAV>, "NumElements", 8
        typeof<D3D12_BUFFER_UAV>, "Flags", 24
        typeof<D3D12_RESOURCE_BARRIER>, "Transition", 8
        typeof<D3D12_RESOURCE_TRANSITION_BARRIER>, "Subresource", 8
        typeof<D3D12_RESOURCE_TRANSITION_BARRIER>, "StateBefore", 12
        typeof<D3D12_RESOURCE_TRANSITION_BARRIER>, "StateAfter", 16
        typeof<D3D12_COMPUTE_PIPELINE_STATE_DESC>, "CS", 8
        typeof<D3D12_DESCRIPTOR_HEAP_DESC>, "NumDescriptors", 4
        typeof<D3D12_DESCRIPTOR_HEAP_DESC>, "Flags", 8
        typeof<D3D12_FEATURE_DATA_D3D12_OPTIONS>, "ResourceBindingTier", 16
        typeof<D3D12_FEATURE_DATA_D3D12_OPTIONS>, "ResourceHeapTier", 56
        typeof<D3D12_FEATURE_DATA_D3D12_OPTIONS1>, "Int64ShaderOps", 20
        typeof<D3D12_FEATURE_DATA_D3D12_OPTIONS4>, "Native16BitShaderOpsSupported", 8
        typeof<D3D12_FEATURE_DATA_D3D12_OPTIONS9>, "AtomicInt64OnTypedResourceSupported", 8
        typeof<D3D12_FEATURE_DATA_D3D12_OPTIONS9>, "WaveMMATier", 20
        typeof<D3D12_DEVICE_CONFIGURATION_DESC>, "SDKVersion", 8
        typeof<DXGI_ADAPTER_DESC1>, "DedicatedVideoMemory", 272
    ]
