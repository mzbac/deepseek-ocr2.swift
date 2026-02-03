import MLX

public enum DeepseekV2Weights {
    public static func sanitizeMoEExpertWeights(
        _ weights: [String: MLXArray],
        config: DeepseekV2Configuration
    ) -> [String: MLXArray] {
        guard let nExperts = config.nRoutedExperts, nExperts > 0 else {
            return weights
        }

        var out = weights

        for layerIdx in 0..<config.numHiddenLayers {
            let isMoELayer =
                config.nRoutedExperts != nil
                && layerIdx >= config.firstKDenseReplace
                && layerIdx % config.moeLayerFreq == 0

            guard isMoELayer else { continue }

            let prefix = "model.layers.\(layerIdx).mlp.experts"

            for proj in ["gate_proj", "up_proj", "down_proj"] {
                let weightKey = "\(prefix).\(proj).weight"
                if out[weightKey] == nil {
                    let perExpert = (0..<nExperts).map { "\(prefix).\($0).\(proj).weight" }
                    if perExpert.allSatisfy({ out[$0] != nil }) {
                        let values = perExpert.map { out.removeValue(forKey: $0)! }
                        out[weightKey] = stacked(values, axis: 0)
                    }
                }

                for suffix in ["scales", "biases"] {
                    let stackedKey = "\(prefix).\(proj).\(suffix)"
                    guard out[stackedKey] == nil else { continue }

                    let perExpert = (0..<nExperts).map { "\(prefix).\($0).\(proj).\(suffix)" }
                    if perExpert.allSatisfy({ out[$0] != nil }) {
                        let values = perExpert.map { out.removeValue(forKey: $0)! }
                        out[stackedKey] = stacked(values, axis: 0)
                    }
                }
            }
        }

        return out
    }
}
