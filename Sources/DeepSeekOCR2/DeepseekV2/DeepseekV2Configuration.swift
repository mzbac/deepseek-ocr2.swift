import Foundation

public struct DeepseekV2Configuration: Codable, Sendable {
    public var vocabSize: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var moeIntermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int

    public var nSharedExperts: Int?
    public var nRoutedExperts: Int?
    public var routedScalingFactor: Float

    public var topkMethod: String
    public var nGroup: Int?
    public var topkGroup: Int?
    public var numExpertsPerTok: Int?
    public var moeLayerFreq: Int
    public var firstKDenseReplace: Int
    public var normTopkProb: Bool
    public var scoringFunc: String

    public var hiddenAct: String
    public var maxPositionEmbeddings: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var attentionBias: Bool
    public var attentionDropout: Float
    public var useMLA: Bool

    public init(
        vocabSize: Int,
        hiddenSize: Int,
        intermediateSize: Int,
        moeIntermediateSize: Int,
        numHiddenLayers: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int? = nil,
        nSharedExperts: Int? = nil,
        nRoutedExperts: Int? = nil,
        routedScalingFactor: Float = 1.0,
        topkMethod: String = "greedy",
        nGroup: Int? = nil,
        topkGroup: Int? = nil,
        numExpertsPerTok: Int? = nil,
        moeLayerFreq: Int = 1,
        firstKDenseReplace: Int = 0,
        normTopkProb: Bool = false,
        scoringFunc: String = "softmax",
        hiddenAct: String = "silu",
        maxPositionEmbeddings: Int = 2048,
        rmsNormEps: Float = 1e-6,
        ropeTheta: Float = 10_000.0,
        attentionBias: Bool = false,
        attentionDropout: Float = 0.0,
        useMLA: Bool = true
    ) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads ?? numAttentionHeads
        self.nSharedExperts = nSharedExperts
        self.nRoutedExperts = nRoutedExperts
        self.routedScalingFactor = routedScalingFactor
        self.topkMethod = topkMethod
        self.nGroup = nGroup
        self.topkGroup = topkGroup
        self.numExpertsPerTok = numExpertsPerTok
        self.moeLayerFreq = moeLayerFreq
        self.firstKDenseReplace = firstKDenseReplace
        self.normTopkProb = normTopkProb
        self.scoringFunc = scoringFunc
        self.hiddenAct = hiddenAct
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.attentionBias = attentionBias
        self.attentionDropout = attentionDropout
        self.useMLA = useMLA
    }

    public enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case topkMethod = "topk_method"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeLayerFreq = "moe_layer_freq"
        case firstKDenseReplace = "first_k_dense_replace"
        case normTopkProb = "norm_topk_prob"
        case scoringFunc = "scoring_func"
        case hiddenAct = "hidden_act"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case useMLA = "use_mla"
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        let hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        let intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        let moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)
        let numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        let numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        let numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? numAttentionHeads

        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads

        self.nSharedExperts = try container.decodeIfPresent(Int.self, forKey: .nSharedExperts)
        self.nRoutedExperts = try container.decodeIfPresent(Int.self, forKey: .nRoutedExperts)
        self.routedScalingFactor = try container.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0

        self.topkMethod = try container.decodeIfPresent(String.self, forKey: .topkMethod) ?? "greedy"
        self.nGroup = try container.decodeIfPresent(Int.self, forKey: .nGroup)
        self.topkGroup = try container.decodeIfPresent(Int.self, forKey: .topkGroup)
        self.numExpertsPerTok = try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok)
        self.moeLayerFreq = try container.decodeIfPresent(Int.self, forKey: .moeLayerFreq) ?? 1
        self.firstKDenseReplace = try container.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 0
        self.normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? false
        self.scoringFunc = try container.decodeIfPresent(String.self, forKey: .scoringFunc) ?? "softmax"

        self.hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        self.maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 2048
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000.0
        self.attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        self.useMLA = try container.decodeIfPresent(Bool.self, forKey: .useMLA) ?? true
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(vocabSize, forKey: .vocabSize)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(moeIntermediateSize, forKey: .moeIntermediateSize)
        try container.encode(numHiddenLayers, forKey: .numHiddenLayers)
        try container.encode(numAttentionHeads, forKey: .numAttentionHeads)
        try container.encode(numKeyValueHeads, forKey: .numKeyValueHeads)
        try container.encodeIfPresent(nSharedExperts, forKey: .nSharedExperts)
        try container.encodeIfPresent(nRoutedExperts, forKey: .nRoutedExperts)
        try container.encode(routedScalingFactor, forKey: .routedScalingFactor)
        try container.encode(topkMethod, forKey: .topkMethod)
        try container.encodeIfPresent(nGroup, forKey: .nGroup)
        try container.encodeIfPresent(topkGroup, forKey: .topkGroup)
        try container.encodeIfPresent(numExpertsPerTok, forKey: .numExpertsPerTok)
        try container.encode(moeLayerFreq, forKey: .moeLayerFreq)
        try container.encode(firstKDenseReplace, forKey: .firstKDenseReplace)
        try container.encode(normTopkProb, forKey: .normTopkProb)
        try container.encode(scoringFunc, forKey: .scoringFunc)
        try container.encode(hiddenAct, forKey: .hiddenAct)
        try container.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try container.encode(rmsNormEps, forKey: .rmsNormEps)
        try container.encode(ropeTheta, forKey: .ropeTheta)
        try container.encode(attentionBias, forKey: .attentionBias)
        try container.encode(attentionDropout, forKey: .attentionDropout)
        try container.encode(useMLA, forKey: .useMLA)
    }

    public static func load(from url: URL) throws -> DeepseekV2Configuration {
        struct Container: Decodable {
            var languageConfig: DeepseekV2Configuration?

            enum CodingKeys: String, CodingKey {
                case languageConfig = "language_config"
            }
        }

        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        if let container = try? decoder.decode(Container.self, from: data),
            let languageConfig = container.languageConfig
        {
            return languageConfig
        }
        return try decoder.decode(DeepseekV2Configuration.self, from: data)
    }
}

