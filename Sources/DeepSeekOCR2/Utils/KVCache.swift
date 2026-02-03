import Foundation
import MLX

public protocol KVCache: AnyObject {
    var offset: Int { get }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)
    func reset()
}

public protocol KVCacheRagged: KVCache {
    /// Per-sequence offsets (lengths) for ragged batched decode.
    ///
    /// Shape: `[batch]` (int32).
    var offsets: MLXArray? { get }

    /// Set per-sequence offsets (lengths).
    func setOffsets(_ offsets: [Int])
}

public final class KVCacheSimple: KVCache {
    private var keys: MLXArray?
    private var values: MLXArray?

    public private(set) var offset: Int = 0

    public var step: Int = 256

    public init() {}

    public func slice(batchIndex: Int) -> KVCacheSimple {
        guard let keys, let values else { return KVCacheSimple() }
        precondition(batchIndex >= 0 && batchIndex < keys.dim(0), "batchIndex out of range")

        let cache = KVCacheSimple()
        cache.step = step
        cache.keys = keys[batchIndex..<(batchIndex + 1), 0..., 0..., 0...]
        cache.values = values[batchIndex..<(batchIndex + 1), 0..., 0..., 0...]
        cache.offset = offset
        return cache
    }

    public func reset() {
        keys = nil
        values = nil
        offset = 0
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = offset

        let needsReset: Bool
        if let currentKeys = self.keys {
            needsReset = (previous + keys.dim(2)) > currentKeys.dim(2)
        } else {
            needsReset = true
        }

        if needsReset {
            let batch = keys.dim(0)
            let kvHeads = keys.dim(1)
            let keyHeadDim = keys.dim(3)
            let valueHeadDim = values.dim(3)

            let needed = previous + keys.dim(2)
            let currentCapacity = self.keys?.dim(2) ?? 0
            let grown = max(currentCapacity * 2, needed)
            let steps = (step + grown - 1) / step
            let capacity = steps * step

            let newKeys = MLXArray.zeros([batch, kvHeads, capacity, keyHeadDim], dtype: keys.dtype)
            let newValues = MLXArray.zeros([batch, kvHeads, capacity, valueHeadDim], dtype: values.dtype)

            if let oldKeys = self.keys, let oldValues = self.values, previous > 0 {
                newKeys[0..., 0..., 0..<previous, 0...] = oldKeys[0..., 0..., 0..<previous, 0...]
                newValues[0..., 0..., 0..<previous, 0...] = oldValues[0..., 0..., 0..<previous, 0...]
            }

            self.keys = newKeys
            self.values = newValues
        }

        guard let cachedKeys = self.keys, let cachedValues = self.values else {
            fatalError("KVCacheSimple internal allocation failed")
        }

        let length = keys.dim(2)
        cachedKeys[0..., 0..., previous..<(previous + length), 0...] = keys
        cachedValues[0..., 0..., previous..<(previous + length), 0...] = values

        self.keys = cachedKeys
        self.values = cachedValues
        offset = previous + length

        return (cachedKeys[0..., 0..., 0..<offset, 0...], cachedValues[0..., 0..., 0..<offset, 0...])
    }
}

/// A KV cache that supports ragged batched decode by maintaining per-sequence offsets.
///
/// - Keys/values are stored in a shared tensor `[B, kvHeads, capacity, headDim]`.
/// - `offsets` tracks the current sequence length per batch element.
/// - For prompt fills, this behaves like an append cache (all sequences share the same prompt length).
/// - For incremental decode (`seqLen == 1`), updates are scattered at `offsets` for each batch element.
public final class KVCacheRaggedSimple: KVCacheRagged {
    private var keys: MLXArray?
    private var values: MLXArray?
    private var offsetsArray: MLXArray?

    nonisolated(unsafe) private static let oneInt32 = MLXArray(Int32(1))

    public private(set) var offset: Int = 0
    public var step: Int = 256

    public init() {}

    public var offsets: MLXArray? {
        offsetsArray
    }

    public func setOffsets(_ offsets: [Int]) {
        guard !offsets.isEmpty else {
            offsetsArray = nil
            offset = 0
            return
        }

        let int32 = offsets.map { Int32($0) }
        offsetsArray = MLXArray(int32)
        offset = offsets.max() ?? 0
    }

    public func reset() {
        keys = nil
        values = nil
        offsetsArray = nil
        offset = 0
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let batch = keys.dim(0)
        let kvHeads = keys.dim(1)
        let length = keys.dim(2)
        let keyHeadDim = keys.dim(3)
        let valueHeadDim = values.dim(3)

        if self.keys == nil || self.values == nil {
            let steps = (step + length - 1) / step
            let capacity = max(steps * step, length)

            let newKeys = MLXArray.zeros([batch, kvHeads, capacity, keyHeadDim], dtype: keys.dtype)
            let newValues = MLXArray.zeros([batch, kvHeads, capacity, valueHeadDim], dtype: values.dtype)

            newKeys[0..., 0..., 0..<length, 0...] = keys
            newValues[0..., 0..., 0..<length, 0...] = values

            self.keys = newKeys
            self.values = newValues
            self.offsetsArray = broadcast(MLXArray(Int32(length)), to: [batch])
            self.offset = length

            return (newKeys[0..., 0..., 0..<length, 0...], newValues[0..., 0..., 0..<length, 0...])
        }

        guard var cachedKeys = self.keys, var cachedValues = self.values else {
            fatalError("KVCacheRaggedSimple internal allocation failed")
        }
        var offsetsArray = self.offsetsArray ?? broadcast(MLXArray(Int32(offset)), to: [batch])
        self.offsetsArray = offsetsArray

        precondition(length == 1, "KVCacheRaggedSimple currently supports ragged updates only for seqLen == 1 (got seqLen=\(length)).")

        let needed = offset + 1
        let currentCapacity = cachedKeys.dim(2)
        if needed > currentCapacity {
            let grown = max(currentCapacity * 2, needed)
            let steps = (step + grown - 1) / step
            let capacity = steps * step

            let newKeys = MLXArray.zeros([batch, kvHeads, capacity, keyHeadDim], dtype: cachedKeys.dtype)
            let newValues = MLXArray.zeros([batch, kvHeads, capacity, valueHeadDim], dtype: cachedValues.dtype)

            if offset > 0 {
                newKeys[0..., 0..., 0..<offset, 0...] = cachedKeys[0..., 0..., 0..<offset, 0...]
                newValues[0..., 0..., 0..<offset, 0...] = cachedValues[0..., 0..., 0..<offset, 0...]
            }

            cachedKeys = newKeys
            cachedValues = newValues
        }

        let positions = broadcast(offsetsArray.reshaped(batch, 1, 1, 1), to: [batch, kvHeads, 1, keyHeadDim])
        cachedKeys = putAlong(cachedKeys, positions, values: keys, axis: 2)

        let valuePositions = broadcast(offsetsArray.reshaped(batch, 1, 1, 1), to: [batch, kvHeads, 1, valueHeadDim])
        cachedValues = putAlong(cachedValues, valuePositions, values: values, axis: 2)

        offsetsArray = offsetsArray + Self.oneInt32
        self.offsetsArray = offsetsArray
        offset += 1

        self.keys = cachedKeys
        self.values = cachedValues

        return (cachedKeys[0..., 0..., 0..<offset, 0...], cachedValues[0..., 0..., 0..<offset, 0...])
    }

    func selecting(batchIndices: [Int]) -> KVCacheRaggedSimple {
        let selected = KVCacheRaggedSimple()
        selected.step = step

        guard !batchIndices.isEmpty else { return selected }
        let int32 = batchIndices.map { Int32($0) }
        let indexArray = MLXArray(int32).asType(.int32)
        let newBatch = batchIndices.count

        if let keys = self.keys, let values = self.values {
            precondition(newBatch <= keys.dim(0), "batchIndices larger than cache batch")

            let keyIdx = broadcast(
                indexArray.reshaped(newBatch, 1, 1, 1),
                to: [newBatch, keys.dim(1), keys.dim(2), keys.dim(3)]
            )
            let valueIdx = broadcast(
                indexArray.reshaped(newBatch, 1, 1, 1),
                to: [newBatch, values.dim(1), values.dim(2), values.dim(3)]
            )

            selected.keys = takeAlong(keys, keyIdx, axis: 0)
            selected.values = takeAlong(values, valueIdx, axis: 0)
            selected.offset = offset
        }

        if let offsetsArray {
            selected.offsetsArray = takeAlong(offsetsArray, indexArray, axis: 0).asType(.int32)
            selected.offset = offset
        }

        return selected
    }
}

public func createCausalMask(n: Int, offset: Int) -> MLXArray {
    var rinds = MLXArray(Int32(0) ..< Int32(offset + n))
    var linds = offset != 0 ? MLXArray(Int32(offset) ..< Int32(offset + n)) : rinds
    linds = linds[0..., .newAxis]
    rinds = rinds[.newAxis]
    return linds .>= rinds
}

private enum RaggedDecodeMask {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var cachedPositions: MLXArray?
    nonisolated(unsafe) private static var cachedCapacity: Int = 0
    private static let step: Int = 256

    private static let compiled = compile(shapeless: true) { (lengths: MLXArray, positions: MLXArray) -> MLXArray in
        // lengths: [batch] (int32)
        // positions: [maxLength] (int32)
        let pos2 = positions.reshaped(1, -1)
        let perBatch = lengths.reshaped(-1, 1)
        let allowed = pos2 .< perBatch
        return allowed.expandedDimensions(axis: 1).expandedDimensions(axis: 1)
    }

    private static func roundUp(_ value: Int, multiple: Int) -> Int {
        guard multiple > 0 else { return value }
        return ((value + multiple - 1) / multiple) * multiple
    }

    private static func positions(upTo length: Int) -> MLXArray {
        lock.lock()
        defer { lock.unlock() }

        let length = max(0, length)
        guard length > 0 else {
            return MLXArray(Int32(0) ..< Int32(0))
        }

        if let cachedPositions, cachedCapacity >= length {
            return cachedPositions[0..<length]
        }

        let grown = max(length, max(step, cachedCapacity * 2))
        let capacity = roundUp(grown, multiple: step)

        let newPositions = MLXArray(Int32(0) ..< Int32(capacity))
        cachedPositions = newPositions
        cachedCapacity = capacity
        return newPositions[0..<length]
    }

    static func make(lengths: MLXArray, maxLength: Int) -> MLXArray {
        let positions = positions(upTo: maxLength)
        return compiled(lengths, positions)
    }
}

public func createRaggedDecodeMask(lengths: MLXArray, maxLength: Int) -> MLXArray {
    precondition(lengths.ndim == 1, "createRaggedDecodeMask expects lengths [batch]")
    precondition(maxLength >= 0, "createRaggedDecodeMask expects maxLength >= 0 (got maxLength=\(maxLength))")

    if maxLength == 0 {
        return MLXArray.zeros([lengths.dim(0), 1, 1, 0], dtype: .bool)
    }

    return RaggedDecodeMask.make(lengths: lengths, maxLength: maxLength)
}
