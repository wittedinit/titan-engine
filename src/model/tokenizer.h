#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace titan {

// ============================================================================
// BPE Tokenizer — Loads HuggingFace tokenizer.json
//
// Supports byte-level BPE (used by Llama, Mistral, Qwen, DeepSeek, etc.)
// ============================================================================

class Tokenizer {
public:
    Tokenizer() = default;
    ~Tokenizer() = default;

    // Load from HuggingFace model directory (reads tokenizer.json + config)
    bool load(const std::string& model_dir);

    // Encode text to token IDs
    std::vector<int> encode(const std::string& text, bool add_bos = true) const;

    // Decode single token ID to string
    std::string decode(int token_id) const;

    // Decode multiple token IDs
    std::string decode(const std::vector<int>& tokens) const;

    // Special token IDs
    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }
    int pad_id() const { return pad_id_; }
    int vocab_size() const { return (int)id_to_token_.size(); }

    // Check if a token is special (BOS, EOS, etc.)
    bool is_special(int token_id) const;

    // Look up token ID by string (returns -1 if not found)
    int token_to_id(const std::string& token) const {
        auto it = token_to_id_.find(token);
        return (it != token_to_id_.end()) ? it->second : -1;
    }

private:
    // Vocabulary: token string ↔ ID
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;

    // BPE merge rules: (token_a, token_b) → merged_token, ordered by priority
    struct MergeRule {
        std::string a;
        std::string b;
        std::string merged;
        int rank;
    };
    std::vector<MergeRule> merges_;
    // Fast lookup: "a b" → rank
    std::unordered_map<std::string, int> merge_ranks_;

    // Byte-level encoding table (byte value → unicode token string)
    std::string byte_to_token_[256];
    std::unordered_map<std::string, uint8_t> token_to_byte_;

    // Special tokens
    int bos_id_ = -1;
    int eos_id_ = -1;
    int pad_id_ = -1;
    std::unordered_map<int, bool> special_ids_;

    // Format flag
    bool is_tiktoken_ = false;

    // Internal BPE methods
    void init_byte_encoding();
    std::vector<std::string> pre_tokenize(const std::string& text) const;
    std::vector<std::string> bpe_encode_word(const std::string& word) const;
    bool parse_tokenizer_json(const std::string& path);
    bool parse_tiktoken(const std::string& tiktoken_path, const std::string& tokenizer_json_path);
    std::vector<int> encode_tiktoken(const std::string& text, bool add_bos) const;
    std::vector<int> tiktoken_bpe_word(const std::string& bytes) const;
    bool parse_special_tokens(const std::string& model_dir);
};

} // namespace titan
