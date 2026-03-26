#include "model/tokenizer.h"
#include "core/logging.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <cstring>

namespace titan {

// ============================================================================
// Byte-Level Encoding (GPT-2 / Llama style)
//
// Maps each byte 0-255 to a unicode character. This avoids special handling
// of whitespace, control chars, etc. — everything is a "visible" token.
// ============================================================================

void Tokenizer::init_byte_encoding() {
    // GPT-2 byte-level mapping:
    // Printable ASCII (33-126) maps to itself
    // Other bytes map to unicode range 256+
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
            byte_to_token_[b] = std::string(1, (char)b);
        } else {
            // Map to unicode range starting at 256
            // Values 256-511 need proper UTF-8 encoding (2 bytes)
            unsigned int codepoint = 256 + n;
            std::string s;
            s += (char)(0xC0 | (codepoint >> 6));
            s += (char)(0x80 | (codepoint & 0x3F));
            byte_to_token_[b] = s;
            n++;
        }
    }

    // Build reverse mapping
    for (int b = 0; b < 256; b++) {
        token_to_byte_[byte_to_token_[b]] = (uint8_t)b;
    }
}

// ============================================================================
// Minimal JSON Helpers (no external dependency)
// ============================================================================

// Find the value of a JSON string key, handling nested structures
static std::string json_find_key(const std::string& json, const std::string& key,
                                  size_t start = 0) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search, start);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return "";
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n'))
        pos++;
    return json.substr(pos);
}

// Extract a JSON string value (handles escape sequences)
static std::string json_extract_string(const std::string& json, size_t& pos) {
    if (pos >= json.size() || json[pos] != '"') return "";
    pos++; // skip opening quote

    std::string result;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            pos++;
            switch (json[pos]) {
                case '"': result += '"'; break;
                case '\\': result += '\\'; break;
                case '/': result += '/'; break;
                case 'n': result += '\n'; break;
                case 'r': result += '\r'; break;
                case 't': result += '\t'; break;
                case 'u': {
                    // Unicode escape \uXXXX
                    if (pos + 4 < json.size()) {
                        std::string hex = json.substr(pos + 1, 4);
                        unsigned int cp = std::stoul(hex, nullptr, 16);
                        if (cp < 0x80) {
                            result += (char)cp;
                        } else if (cp < 0x800) {
                            result += (char)(0xC0 | (cp >> 6));
                            result += (char)(0x80 | (cp & 0x3F));
                        } else {
                            result += (char)(0xE0 | (cp >> 12));
                            result += (char)(0x80 | ((cp >> 6) & 0x3F));
                            result += (char)(0x80 | (cp & 0x3F));
                        }
                        pos += 4;
                    }
                    break;
                }
                default: result += json[pos]; break;
            }
        } else {
            result += json[pos];
        }
        pos++;
    }
    if (pos < json.size()) pos++; // skip closing quote
    return result;
}

// ============================================================================
// Parse tokenizer.json
// ============================================================================

bool Tokenizer::parse_tokenizer_json(const std::string& path) {
    std::ifstream f(path);
    if (!f.good()) {
        LOG_ERROR("Cannot open tokenizer: %s", path.c_str());
        return false;
    }

    std::string json((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());

    // Parse vocabulary from "model.vocab" section
    // Format: {"model": {"vocab": {"token": id, ...}, "merges": ["a b", ...]}}
    auto vocab_pos = json.find("\"vocab\"");
    if (vocab_pos == std::string::npos) {
        LOG_ERROR("No vocab section in tokenizer.json");
        return false;
    }

    // Find the opening brace of vocab object
    auto brace_start = json.find('{', vocab_pos + 7);
    if (brace_start == std::string::npos) return false;

    // Parse vocab entries: "token_string": integer_id
    size_t pos = brace_start + 1;
    int max_id = -1;

    while (pos < json.size()) {
        // Skip whitespace
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' ||
               json[pos] == '\r' || json[pos] == '\t' || json[pos] == ','))
            pos++;

        if (pos >= json.size() || json[pos] == '}') break;

        // Parse token string
        std::string token = json_extract_string(json, pos);
        if (token.empty() && json[pos] != '"') break;

        // Skip to colon
        while (pos < json.size() && json[pos] != ':') pos++;
        pos++; // skip colon

        // Skip whitespace
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

        // Parse integer ID
        size_t id_start = pos;
        while (pos < json.size() && (json[pos] >= '0' && json[pos] <= '9')) pos++;

        if (pos > id_start) {
            int id = std::stoi(json.substr(id_start, pos - id_start));
            token_to_id_[token] = id;
            if (id > max_id) max_id = id;
        }
    }

    // Build ID-to-token lookup
    if (max_id >= 0) {
        id_to_token_.resize(max_id + 1);
        for (const auto& [token, id] : token_to_id_) {
            if (id >= 0 && id <= max_id) {
                id_to_token_[id] = token;
            }
        }
    }

    LOG_INFO("Loaded vocabulary: %zu tokens (max_id=%d)", token_to_id_.size(), max_id);

    // Parse merges from "model.merges" section
    auto merges_pos = json.find("\"merges\"");
    if (merges_pos != std::string::npos) {
        auto bracket_start = json.find('[', merges_pos);
        if (bracket_start != std::string::npos) {
            pos = bracket_start + 1;
            int rank = 0;

            while (pos < json.size()) {
                while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' ||
                       json[pos] == '\r' || json[pos] == '\t' || json[pos] == ','))
                    pos++;

                if (pos >= json.size() || json[pos] == ']') break;

                std::string merge_str = json_extract_string(json, pos);
                if (merge_str.empty()) break;

                // Split "token_a token_b" on first space
                auto space = merge_str.find(' ');
                if (space != std::string::npos) {
                    MergeRule rule;
                    rule.a = merge_str.substr(0, space);
                    rule.b = merge_str.substr(space + 1);
                    rule.merged = rule.a + rule.b;
                    rule.rank = rank;
                    merges_.push_back(rule);
                    merge_ranks_[merge_str] = rank;
                    rank++;
                }
            }
        }
    }

    LOG_INFO("Loaded %zu BPE merge rules", merges_.size());
    return true;
}

bool Tokenizer::parse_special_tokens(const std::string& model_dir) {
    // Try tokenizer_config.json for special tokens
    std::string config_path = model_dir + "/tokenizer_config.json";
    std::ifstream f(config_path);
    if (!f.good()) {
        // Try special_tokens_map.json
        config_path = model_dir + "/special_tokens_map.json";
        f.open(config_path);
    }

    if (f.good()) {
        std::string json((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());

        // Look for bos_token, eos_token
        auto find_special = [&](const std::string& key) -> int {
            auto val = json_find_key(json, key);
            if (val.empty()) return -1;

            // Could be a string or an object with "content" field
            size_t p = 0;
            while (p < val.size() && val[p] != '"') p++;
            if (p < val.size()) {
                std::string token = json_extract_string(val, p);
                auto it = token_to_id_.find(token);
                if (it != token_to_id_.end()) return it->second;
            }
            return -1;
        };

        bos_id_ = find_special("bos_token");
        eos_id_ = find_special("eos_token");
        pad_id_ = find_special("pad_token");
    }

    // Fallback: common defaults
    if (bos_id_ < 0) {
        // Try common BOS tokens
        for (const auto& t : {"<s>", "<|begin_of_text|>", "<bos>"}) {
            auto it = token_to_id_.find(t);
            if (it != token_to_id_.end()) { bos_id_ = it->second; break; }
        }
    }
    if (eos_id_ < 0) {
        for (const auto& t : {"</s>", "<|end_of_text|>", "<|eot_id|>", "<eos>"}) {
            auto it = token_to_id_.find(t);
            if (it != token_to_id_.end()) { eos_id_ = it->second; break; }
        }
    }

    // Mark special tokens
    if (bos_id_ >= 0) special_ids_[bos_id_] = true;
    if (eos_id_ >= 0) special_ids_[eos_id_] = true;
    if (pad_id_ >= 0) special_ids_[pad_id_] = true;

    // Also mark added_tokens from tokenizer.json as special
    // (usually control tokens like <|im_start|>, <|im_end|>, etc.)

    LOG_INFO("Special tokens: BOS=%d EOS=%d PAD=%d", bos_id_, eos_id_, pad_id_);
    return true;
}

// ============================================================================
// Load
// ============================================================================

bool Tokenizer::load(const std::string& model_dir) {
    init_byte_encoding();

    std::string tok_path = model_dir + "/tokenizer.json";
    if (!parse_tokenizer_json(tok_path)) {
        LOG_ERROR("Failed to parse tokenizer.json");
        return false;
    }

    parse_special_tokens(model_dir);
    return true;
}

// ============================================================================
// Pre-tokenization (split text into words for BPE)
// ============================================================================

std::vector<std::string> Tokenizer::pre_tokenize(const std::string& text) const {
    // GPT-2 / Llama pre-tokenization pattern:
    // Split on whitespace boundaries, keeping the space attached to the following word
    // Also split on punctuation
    std::vector<std::string> words;
    std::string current;

    for (size_t i = 0; i < text.size(); i++) {
        char c = text[i];

        // Space starts a new word (Llama uses \u2581 / byte 0xE2 0x96 0x81 for space)
        if (c == ' ') {
            if (!current.empty()) {
                words.push_back(current);
                current.clear();
            }
            // Convert space to the byte-encoded representation
            current += byte_to_token_[(uint8_t)' '];
        } else if (c == '\n' || c == '\r' || c == '\t') {
            if (!current.empty()) {
                words.push_back(current);
                current.clear();
            }
            // Encode control characters
            current += byte_to_token_[(uint8_t)c];
            words.push_back(current);
            current.clear();
        } else {
            // Regular character — encode to byte-level token
            current += byte_to_token_[(uint8_t)c];
        }
    }

    if (!current.empty()) {
        words.push_back(current);
    }

    return words;
}

// ============================================================================
// BPE Encoding (apply merge rules to a single word)
// ============================================================================

std::vector<std::string> Tokenizer::bpe_encode_word(const std::string& word) const {
    if (word.empty()) return {};

    // Start with each character as a separate token
    std::vector<std::string> tokens;
    for (size_t i = 0; i < word.size(); ) {
        // Handle multi-byte UTF-8 characters
        int len = 1;
        uint8_t c = (uint8_t)word[i];
        if (c >= 0xC0 && c < 0xE0) len = 2;
        else if (c >= 0xE0 && c < 0xF0) len = 3;
        else if (c >= 0xF0) len = 4;
        len = std::min(len, (int)(word.size() - i));
        tokens.push_back(word.substr(i, len));
        i += len;
    }

    if (tokens.size() <= 1) return tokens;

    // Iteratively apply the highest-priority merge
    while (tokens.size() > 1) {
        // Find the pair with the lowest rank (highest priority)
        int best_rank = INT_MAX;
        int best_idx = -1;

        for (int i = 0; i < (int)tokens.size() - 1; i++) {
            std::string pair = tokens[i] + " " + tokens[i + 1];
            auto it = merge_ranks_.find(pair);
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }

        if (best_idx < 0) break; // No more applicable merges

        // Apply the merge
        tokens[best_idx] = tokens[best_idx] + tokens[best_idx + 1];
        tokens.erase(tokens.begin() + best_idx + 1);
    }

    return tokens;
}

// ============================================================================
// Encode
// ============================================================================

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int> result;

    if (add_bos && bos_id_ >= 0) {
        result.push_back(bos_id_);
    }

    // Pre-tokenize into words
    auto words = pre_tokenize(text);

    // BPE encode each word
    for (const auto& word : words) {
        auto tokens = bpe_encode_word(word);

        for (const auto& tok : tokens) {
            auto it = token_to_id_.find(tok);
            if (it != token_to_id_.end()) {
                result.push_back(it->second);
            } else {
                // Unknown token — encode as individual bytes
                for (char c : tok) {
                    auto byte_tok = byte_to_token_[(uint8_t)c];
                    auto it2 = token_to_id_.find(byte_tok);
                    if (it2 != token_to_id_.end()) {
                        result.push_back(it2->second);
                    } else {
                        LOG_WARN("Unknown byte token: 0x%02X", (uint8_t)c);
                    }
                }
            }
        }
    }

    return result;
}

// ============================================================================
// Decode
// ============================================================================

std::string Tokenizer::decode(int token_id) const {
    if (token_id < 0 || token_id >= (int)id_to_token_.size()) return "";
    if (is_special(token_id)) return ""; // Don't decode special tokens as text

    const std::string& tok = id_to_token_[token_id];

    // Convert byte-level tokens back to actual bytes
    std::string result;
    for (size_t i = 0; i < tok.size(); ) {
        int len = 1;
        uint8_t c = (uint8_t)tok[i];
        if (c >= 0xC0 && c < 0xE0) len = 2;
        else if (c >= 0xE0 && c < 0xF0) len = 3;
        else if (c >= 0xF0) len = 4;
        len = std::min(len, (int)(tok.size() - i));

        std::string ch = tok.substr(i, len);
        auto it = token_to_byte_.find(ch);
        if (it != token_to_byte_.end()) {
            result += (char)it->second;
        } else {
            result += ch; // Pass through if not in byte map
        }
        i += len;
    }

    return result;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    std::string result;
    for (int id : tokens) {
        result += decode(id);
    }
    return result;
}

bool Tokenizer::is_special(int token_id) const {
    return special_ids_.count(token_id) > 0;
}

} // namespace titan
