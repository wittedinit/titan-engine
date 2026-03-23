// Python bindings for Titan Engine via pybind11
//
// Install: pip install pybind11
// Build: cmake .. -DTITAN_BUILD_PYTHON=ON
//
// Usage:
//   import titan
//   engine = titan.Engine()
//   engine.load("/path/to/model", quant="q4_k")
//   for token in engine.generate("Hello!", max_tokens=100):
//       print(token, end="", flush=True)

#ifdef TITAN_BUILD_PYTHON

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "inference/engine.h"
#include "core/types.h"
#include "core/hardware.h"
#include "core/logging.h"

namespace py = pybind11;

using namespace titan;

// Python-friendly wrapper around InferenceEngine
class PyEngine {
public:
    PyEngine() = default;

    void load(const std::string& model_path,
              const std::string& quant = "q4_k",
              int context = 8192,
              int vram_mb = 0,
              int ram_mb = 0) {
        RuntimeConfig cfg;
        cfg.model_path = model_path;
        cfg.max_context_len = context;
        cfg.vram_budget_mb = vram_mb;
        cfg.ram_budget_mb = ram_mb;

        if (quant == "fp16") cfg.weight_dtype = DType::FP16;
        else if (quant == "fp8") cfg.weight_dtype = DType::FP8_E4M3;
        else if (quant == "int4" || quant == "q4") cfg.weight_dtype = DType::INT4;
        else if (quant == "q4_k") cfg.weight_dtype = DType::Q4_K;
        else if (quant == "q3_k") cfg.weight_dtype = DType::Q3_K;
        else if (quant == "int2") cfg.weight_dtype = DType::INT2;
        else if (quant == "fp4") cfg.weight_dtype = DType::FP4;

        if (!engine_.initialize(cfg)) {
            throw std::runtime_error("Failed to initialize engine");
        }
        if (!engine_.load_model(model_path)) {
            throw std::runtime_error("Failed to load model from " + model_path);
        }
        loaded_ = true;
    }

    // Generate with callback (streaming)
    std::string generate(const std::string& prompt,
                         float temperature = 0.7f,
                         float top_p = 0.9f,
                         int top_k = 40,
                         int max_tokens = 2048,
                         py::object callback = py::none()) {
        if (!loaded_) throw std::runtime_error("No model loaded");

        SamplingParams sampling;
        sampling.temperature = temperature;
        sampling.top_p = top_p;
        sampling.top_k = top_k;
        sampling.max_tokens = max_tokens;

        std::string result;
        bool has_callback = !callback.is_none();

        engine_.generate(prompt, sampling,
            [&](int token_id, const std::string& text) {
                result += text;
                if (has_callback) {
                    py::gil_scoped_acquire acquire;
                    callback(text);
                }
            });

        return result;
    }

    // Chat-style generation (OpenAI-compatible message format)
    std::string chat(const std::vector<std::pair<std::string, std::string>>& messages,
                     float temperature = 0.7f,
                     int max_tokens = 2048) {
        // Build prompt from messages
        std::string prompt;
        for (const auto& [role, content] : messages) {
            if (role == "system") {
                prompt += "System: " + content + "\n\n";
            } else if (role == "user") {
                prompt += "User: " + content + "\n\n";
            } else if (role == "assistant") {
                prompt += "Assistant: " + content + "\n\n";
            }
        }
        prompt += "Assistant: ";

        return generate(prompt, temperature, 0.9f, 40, max_tokens);
    }

    // Model info
    std::string model_name() const {
        return loaded_ ? engine_.model_config().name : "";
    }
    size_t total_params() const {
        return loaded_ ? engine_.model_config().total_params() : 0;
    }
    size_t active_params() const {
        return loaded_ ? engine_.model_config().active_params_per_token() : 0;
    }
    int vocab_size() const {
        return loaded_ ? engine_.model_config().vocab_size : 0;
    }

    // Hardware info
    static py::dict hardware_info() {
        auto hw = detect_hardware();
        py::dict info;
        if (!hw.gpus.empty()) {
            info["gpu_name"] = hw.gpus[0].name;
            info["gpu_vram_gb"] = hw.gpus[0].vram_total / 1e9;
            info["gpu_vram_free_gb"] = hw.gpus[0].vram_free / 1e9;
            info["gpu_compute"] = std::to_string(hw.gpus[0].compute_cap_major) + "." +
                                  std::to_string(hw.gpus[0].compute_cap_minor);
        }
        info["cpu_name"] = hw.cpu.model_name;
        info["cpu_cores"] = hw.cpu.logical_cores;
        info["ram_total_gb"] = hw.memory.total_ram / 1e9;
        info["ram_available_gb"] = hw.memory.available_ram / 1e9;
        info["has_avx512"] = hw.cpu.has_avx512f;
        return info;
    }

    // Encode/decode tokens
    std::vector<int> encode(const std::string& text) {
        if (!loaded_) throw std::runtime_error("No model loaded");
        return engine_.tokenizer().encode(text);
    }
    std::string decode(const std::vector<int>& tokens) {
        if (!loaded_) throw std::runtime_error("No model loaded");
        return engine_.tokenizer().decode(tokens);
    }

private:
    InferenceEngine engine_;
    bool loaded_ = false;
};

PYBIND11_MODULE(titan, m) {
    m.doc() = "Titan Engine — Universal LLM Inference Engine";

    py::class_<PyEngine>(m, "Engine")
        .def(py::init<>())
        .def("load", &PyEngine::load,
             py::arg("model_path"),
             py::arg("quant") = "q4_k",
             py::arg("context") = 8192,
             py::arg("vram_mb") = 0,
             py::arg("ram_mb") = 0,
             "Load a model from a HuggingFace directory or GGUF file")
        .def("generate", &PyEngine::generate,
             py::arg("prompt"),
             py::arg("temperature") = 0.7f,
             py::arg("top_p") = 0.9f,
             py::arg("top_k") = 40,
             py::arg("max_tokens") = 2048,
             py::arg("callback") = py::none(),
             "Generate text from a prompt. Optional callback for streaming.")
        .def("chat", &PyEngine::chat,
             py::arg("messages"),
             py::arg("temperature") = 0.7f,
             py::arg("max_tokens") = 2048,
             "Chat-style generation from a list of (role, content) tuples")
        .def("encode", &PyEngine::encode, "Tokenize text to token IDs")
        .def("decode", &PyEngine::decode, "Decode token IDs to text")
        .def_property_readonly("model_name", &PyEngine::model_name)
        .def_property_readonly("total_params", &PyEngine::total_params)
        .def_property_readonly("active_params", &PyEngine::active_params)
        .def_property_readonly("vocab_size", &PyEngine::vocab_size)
        .def_static("hardware_info", &PyEngine::hardware_info,
                     "Detect and return hardware capabilities");

    // Convenience function
    m.def("hardware", &PyEngine::hardware_info,
          "Detect and return hardware capabilities");
}

#endif // TITAN_BUILD_PYTHON
