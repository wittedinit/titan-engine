#include "model/tokenizer.h"
#include "core/logging.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <cstring>
#include <climits>

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
        LOG_WARN("No vocab section in tokenizer.json — will try tiktoken format");
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
        config_path = model_dir + "/special_tokens_map.json";
        f.open(config_path);
    }

    if (f.good()) {
        std::string json((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());

        // Helper: find a special token ID.
        // Handles both string values ("token_str" looked up in vocab)
        // and direct numeric values (e.g. bos_token_id: 163584).
        auto find_special = [&](const std::string& key) -> int {
            // Try numeric key variant first (e.g. "bos_token_id")
            auto num_val = json_find_key(json, key + "_id");
            if (!num_val.empty()) {
                size_t p = 0;
                while (p < num_val.size() && (num_val[p] < '0' || num_val[p] > '9') && num_val[p] != '-') p++;
                if (p < num_val.size() && (num_val[p] >= '0' && num_val[p] <= '9')) {
                    try { return std::stoi(num_val.substr(p)); } catch (...) {}
                }
            }
            // Try string value
            auto val = json_find_key(json, key);
            if (val.empty()) return -1;
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

    // Fallback: look up common token strings in the vocab
    if (bos_id_ < 0) {
        for (const auto& t : {"<s>", "<|begin_of_text|>", "<bos>", "<|im_start|>"}) {
            auto it = token_to_id_.find(t);
            if (it != token_to_id_.end()) { bos_id_ = it->second; break; }
        }
    }
    if (eos_id_ < 0) {
        for (const auto& t : {"</s>", "<|end_of_text|>", "<|eot_id|>", "<eos>", "<|im_end|>"}) {
            auto it = token_to_id_.find(t);
            if (it != token_to_id_.end()) { eos_id_ = it->second; break; }
        }
    }

    // Mark all added tokens as special
    if (bos_id_ >= 0) special_ids_[bos_id_] = true;
    if (eos_id_ >= 0) special_ids_[eos_id_] = true;
    if (pad_id_ >= 0) special_ids_[pad_id_] = true;

    // Mark any token whose string starts with '<|' as special
    for (const auto& [tok, id] : token_to_id_) {
        if (tok.size() >= 2 && tok[0] == '<' && tok[1] == '|') {
            special_ids_[id] = true;
        }
    }

    LOG_INFO("Special tokens: BOS=%d EOS=%d PAD=%d", bos_id_, eos_id_, pad_id_);
    return true;
}

// ============================================================================
// Base64 decode (RFC 4648)
// ============================================================================

static std::string base64_decode(const std::string& encoded) {
    static const uint8_t table[256] = {
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,62,64,64,64,63,
        52,53,54,55,56,57,58,59,60,61,64,64,64,64,64,64,
        64, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
        15,16,17,18,19,20,21,22,23,24,25,64,64,64,64,64,
        64,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,51,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
    };
    std::string out;
    out.reserve(encoded.size() * 3 / 4);
    uint32_t val = 0;
    int bits = 0;
    for (unsigned char c : encoded) {
        uint8_t b = table[c];
        if (b == 64) continue;  // padding or invalid
        val = (val << 6) | b;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out += (char)((val >> bits) & 0xFF);
        }
    }
    return out;
}

// ============================================================================
// Parse added_tokens_decoder from tokenizer.json (or tokenizer_config.json)
// Format: "added_tokens_decoder": { "163584": {"content": "[BOS]", ...}, ... }
// Adds special tokens (BOS, EOS, etc.) to the vocab maps.
// ============================================================================

static void parse_added_tokens_json(const std::string& json,
                                    std::unordered_map<std::string, int>& token_to_id,
                                    int& max_id) {
    auto adt_pos = json.find("\"added_tokens_decoder\"");
    if (adt_pos == std::string::npos) return;

    // Find the opening { of the outer decoder object
    auto obj_start = json.find('{', adt_pos + 22);
    if (obj_start == std::string::npos) return;

    size_t pos = obj_start + 1;
    while (pos < json.size()) {
        // Skip whitespace/commas to next key or end-of-object
        while (pos < json.size() && json[pos] != '"' && json[pos] != '}') pos++;
        if (pos >= json.size() || json[pos] == '}') break;

        // Parse the numeric key (token ID as string)
        std::string id_str = json_extract_string(json, pos);
        bool is_num = !id_str.empty() &&
                      std::all_of(id_str.begin(), id_str.end(), ::isdigit);
        if (!is_num) {
            // Not a numeric key — skip past colon + value
            while (pos < json.size() && json[pos] != ':') pos++;
            if (pos < json.size()) pos++;
            // Skip value (could be string, number, object, array)
            while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\t')) pos++;
            if (pos < json.size() && json[pos] == '{') {
                int d = 1; pos++;
                while (pos < json.size() && d > 0) {
                    if (json[pos] == '{') d++;
                    else if (json[pos] == '}') d--;
                    pos++;
                }
            }
            continue;
        }

        int token_id = std::stoi(id_str);

        // Skip to the inner object
        while (pos < json.size() && json[pos] != ':') pos++;
        if (pos < json.size()) pos++;
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\t')) pos++;
        if (pos >= json.size() || json[pos] != '{') continue;
        size_t inner_start = pos;
        pos++; // enter inner object

        // Find "content" key in the inner object
        std::string content;
        while (pos < json.size() && json[pos] != '}') {
            while (pos < json.size() && json[pos] != '"' && json[pos] != '}') pos++;
            if (pos >= json.size() || json[pos] == '}') break;

            std::string key = json_extract_string(json, pos);
            while (pos < json.size() && json[pos] != ':') pos++;
            if (pos < json.size()) pos++;
            while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

            if (key == "content") {
                if (pos < json.size() && json[pos] == '"') {
                    content = json_extract_string(json, pos);
                }
            } else {
                // Skip value
                if (pos < json.size() && json[pos] == '"') {
                    json_extract_string(json, pos);
                } else {
                    while (pos < json.size() && json[pos] != ',' && json[pos] != '}') pos++;
                }
            }
        }
        if (pos < json.size()) pos++; // skip '}'

        if (!content.empty()) {
            token_to_id[content] = token_id;
            if (token_id > max_id) max_id = token_id;
        }
    }
}

// ============================================================================
// Parse tiktoken.model
// Format: one line per token: base64_encoded_bytes<space>rank
// ============================================================================

bool Tokenizer::parse_tiktoken(const std::string& tiktoken_path,
                                const std::string& tokenizer_json_path) {
    std::ifstream f(tiktoken_path);
    if (!f.good()) {
        LOG_ERROR("Cannot open tiktoken model: %s", tiktoken_path.c_str());
        return false;
    }

    int max_id = -1;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto space = line.rfind(' ');
        if (space == std::string::npos) continue;

        const std::string b64 = line.substr(0, space);
        int rank;
        try { rank = std::stoi(line.substr(space + 1)); }
        catch (...) { continue; }

        std::string bytes = base64_decode(b64);
        if (bytes.empty()) continue;

        token_to_id_[bytes] = rank;
        if (rank > max_id) max_id = rank;
    }

    if (token_to_id_.empty()) {
        LOG_ERROR("tiktoken.model parsed 0 tokens");
        return false;
    }

    // Load added_tokens_decoder from tokenizer.json and tokenizer_config.json
    std::string config_json_path = tokenizer_json_path.substr(0, tokenizer_json_path.rfind('/') + 1)
                                   + "tokenizer_config.json";
    for (const auto& path : {tokenizer_json_path, config_json_path}) {
        std::ifstream fj(path);
        if (!fj.good()) continue;
        std::string json((std::istreambuf_iterator<char>(fj)),
                          std::istreambuf_iterator<char>());
        parse_added_tokens_json(json, token_to_id_, max_id);
    }

    // Build id_to_token lookup
    id_to_token_.resize(max_id + 1);
    for (const auto& [bytes, id] : token_to_id_) {
        if (id >= 0 && id <= max_id) {
            id_to_token_[id] = bytes;
        }
    }

    is_tiktoken_ = true;
    LOG_INFO("Loaded tiktoken vocab: %zu tokens (max_id=%d)", token_to_id_.size(), max_id);
    return true;
}

// ============================================================================
// tiktoken BPE: encode a single pre-tokenized piece (raw bytes)
// ============================================================================

std::vector<int> Tokenizer::tiktoken_bpe_word(const std::string& bytes) const {
    if (bytes.empty()) return {};

    // Start: each byte is a separate token
    std::vector<std::string> parts;
    parts.reserve(bytes.size());
    for (unsigned char c : bytes) {
        parts.push_back(std::string(1, (char)c));
    }

    // Iteratively merge the pair whose merged form has the lowest rank
    while (parts.size() > 1) {
        int best_rank = INT_MAX;
        int best_idx = -1;

        for (int i = 0; i < (int)parts.size() - 1; i++) {
            std::string merged = parts[i] + parts[i + 1];
            auto it = token_to_id_.find(merged);
            if (it != token_to_id_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }

        if (best_idx < 0) break; // no more applicable merges

        parts[best_idx] += parts[best_idx + 1];
        parts.erase(parts.begin() + best_idx + 1);
    }

    // Convert to IDs
    std::vector<int> ids;
    ids.reserve(parts.size());
    for (const auto& p : parts) {
        auto it = token_to_id_.find(p);
        if (it != token_to_id_.end()) {
            ids.push_back(it->second);
        } else {
            // Byte-level fallback
            for (unsigned char c : p) {
                std::string b(1, (char)c);
                auto it2 = token_to_id_.find(b);
                if (it2 != token_to_id_.end()) {
                    ids.push_back(it2->second);
                } else {
                    LOG_WARN("tiktoken: unmappable byte 0x%02X", (unsigned)c);
                }
            }
        }
    }
    return ids;
}

// ============================================================================
// tiktoken encode: pre-tokenize + BPE
//
// Uses a simplified regex pattern:
//   - Runs of alphabetic chars (letters)
//   - Runs of digits
//   - Whitespace (preserved as leading space on next word, like GPT-4)
//   - Individual punctuation/other chars
// ============================================================================

std::vector<int> Tokenizer::encode_tiktoken(const std::string& text, bool add_bos) const {
    std::vector<int> result;
    if (add_bos && bos_id_ >= 0) {
        result.push_back(bos_id_);
    }

    if (text.empty()) return result;

    // Pre-tokenize: split into pieces using simple character-class boundaries.
    // The piece includes any leading whitespace (space attaches to the following word).
    std::vector<std::string> pieces;
    size_t i = 0;
    while (i < text.size()) {
        std::string piece;

        // Collect leading whitespace/newlines
        while (i < text.size() && (uint8_t)text[i] <= 0x20) {
            piece += text[i++];
        }
        if (i >= text.size()) {
            if (!piece.empty()) pieces.push_back(piece);
            break;
        }

        unsigned char c = (unsigned char)text[i];
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80) {
            // Letter or multi-byte UTF-8 start — collect whole UTF-8 word
            while (i < text.size()) {
                unsigned char ch = (unsigned char)text[i];
                if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch >= 0x80) {
                    // handle multi-byte
                    int len = 1;
                    if (ch >= 0xF0) len = 4;
                    else if (ch >= 0xE0) len = 3;
                    else if (ch >= 0xC0) len = 2;
                    for (int k = 0; k < len && i < text.size(); k++) piece += text[i++];
                } else {
                    break;
                }
            }
        } else if (c >= '0' && c <= '9') {
            // Digit run
            while (i < text.size() && (unsigned char)text[i] >= '0' && (unsigned char)text[i] <= '9')
                piece += text[i++];
        } else {
            // Single punctuation / other
            piece += text[i++];
        }

        if (!piece.empty()) pieces.push_back(piece);
    }

    // BPE encode each piece
    for (const auto& piece : pieces) {
        auto ids = tiktoken_bpe_word(piece);
        result.insert(result.end(), ids.begin(), ids.end());
    }

    return result;
}

// ============================================================================
// Load
// ============================================================================

bool Tokenizer::load(const std::string& model_dir) {
    init_byte_encoding();

    std::string tok_path = model_dir + "/tokenizer.json";
    std::string tiktoken_path = model_dir + "/tiktoken.model";

    // Try HuggingFace BPE format first
    bool hf_ok = parse_tokenizer_json(tok_path);

    // If HF parse failed or yielded no vocabulary, try tiktoken format
    if (!hf_ok || token_to_id_.empty()) {
        token_to_id_.clear();
        id_to_token_.clear();
        if (!parse_tiktoken(tiktoken_path, tok_path)) {
            LOG_ERROR("Failed to parse tokenizer (tried both tokenizer.json and tiktoken.model)");
            return false;
        }
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
    if (is_tiktoken_) {
        return encode_tiktoken(text, add_bos);
    }

    std::vector<int> result;
    if (add_bos && bos_id_ >= 0) {
        result.push_back(bos_id_);
    }

    auto words = pre_tokenize(text);
    for (const auto& word : words) {
        auto tokens = bpe_encode_word(word);
        for (const auto& tok : tokens) {
            auto it = token_to_id_.find(tok);
            if (it != token_to_id_.end()) {
                result.push_back(it->second);
            } else {
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
    if (is_special(token_id)) return "";

    const std::string& tok = id_to_token_[token_id];

    // tiktoken tokens ARE raw UTF-8 bytes — return directly
    if (is_tiktoken_) return tok;

    // GPT-2 BPE: convert byte-level unicode encoding back to raw bytes
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
            result += ch;
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
