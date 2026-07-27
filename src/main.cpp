#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <string>

#include <storm-parsers/api/explicit_models.h>
#include <storm/api/storm.h>
#include <storm/io/ModelExportFormat.h>
#include <storm/utility/initialize.h>

#include "ruffle.h"

namespace {

// Prints the wallclock time elapsed since construction when it goes out of scope, so it fires on every
// exit path (success or error) without having to duplicate a print call at each return statement.
class RuntimeReporter {
   public:
    RuntimeReporter() : start_(std::chrono::steady_clock::now()) {}

    ~RuntimeReporter() {
        double const seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
        std::cout << "Runtime: " << std::fixed << std::setprecision(3) << seconds << "s.\n";
    }

   private:
    std::chrono::steady_clock::time_point start_;
};

struct Options {
    std::string inputFile;
    std::string outputFile;
    std::string mode;

    double lambda = 0.01;
    std::optional<uint64_t> samples;
    std::optional<double> delta;
    std::optional<uint64_t> seed;
    bool fullCoverage = false;
    std::optional<double> epsilon;
};

void printHelp(char const* programName) {
    std::cout << "ruffle - learns an interval model from a Markov model.\n\n"
              << "Usage: " << programName << " --input <file> --output <file> --mode <mode> [options]\n\n"
              << "Required:\n"
              << "  --input <file>   Input model (.umb, .drn, .drn.gz, .drn.xz; detected from the extension).\n"
              << "  --output <file>  Output model (same formats; may differ from --input).\n"
              << "  --mode <mode>    What to do to the model (see below).\n\n"
              << "Modes:\n"
              << "  learn-interval        Learn an IMDP by sampling the model as a black-box system and computing\n"
              << "                        Clopper-Pearson confidence intervals on the observed probabilities.\n"
              << "                        Requires exactly one of --samples/--delta.\n"
              << "                          --lambda <double>   Local failure probability per successor. Default: 0.01.\n"
              << "                          --samples <uint>    Draw exactly this many samples per state-action pair (e.g. 10000 or 1e4).\n"
              << "                          --delta <double>    Sample until the feasible L1 diameter is at most this.\n"
              << "                          --full-coverage     Keep sampling until every successor with probability > 0\n"
              << "                                              has been sampled at least once.\n"
              << "                          --seed <uint64>     RNG seed. If omitted, one is drawn and printed.\n\n"
              << "  widen-interval        Deterministically widen every probability in (0, 1) into an interval of\n"
              << "                        width --delta, centered at that probability (clamped to [0, 1]). No sampling.\n"
              << "                          --delta <double>    Required. Width of the interval.\n"
              << "                          --epsilon <double>  Raise every lower bound to at least min(p, epsilon).\n\n"
              << "  sample-distribution   Sample the real distribution and replace it by the empirical distribution.\n"
              << "                        Requires exactly one of --samples/--delta.\n"
              << "                          --samples <uint>    Draw exactly this many samples per state-action pair (e.g. 10000 or 1e4).\n"
              << "                          --delta <double>    Sample until the L1 distance to the real distribution\n"
              << "                                              is at most this.\n"
              << "                          --full-coverage, --seed <uint64>   Same as for learn-interval.\n\n"
              << "  --help                Print this help and exit.\n";
}

// Parses a non-negative sample count, accepting scientific notation (e.g. "1e5") as well as plain integers.
uint64_t parseSampleCount(std::string const& text, std::string const& flag) {
    std::size_t pos = 0;
    double parsed;
    try {
        parsed = std::stod(text, &pos);
    } catch (std::exception const&) {
        throw std::runtime_error("Invalid number for " + flag + ": '" + text + "'.");
    }
    if (pos != text.size() || !std::isfinite(parsed) || parsed < 0.0) {
        throw std::runtime_error("Invalid number for " + flag + ": '" + text + "'.");
    }
    return static_cast<uint64_t>(std::llround(parsed));
}

// Parses `--flag value` pairs from argv. Returns std::nullopt (after printing help) if --help was given.
std::optional<Options> parseOptions(int argc, char* argv[]) {
    Options options;

    auto requireValue = [&](int& i, std::string const& flag) -> std::string {
        if (i + 1 >= argc) {
            throw std::runtime_error("Missing value for " + flag + ".");
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printHelp(argv[0]);
            return std::nullopt;
        } else if (arg == "--input") {
            options.inputFile = requireValue(i, arg);
        } else if (arg == "--output") {
            options.outputFile = requireValue(i, arg);
        } else if (arg == "--mode") {
            options.mode = requireValue(i, arg);
        } else if (arg == "--lambda") {
            options.lambda = std::stod(requireValue(i, arg));
        } else if (arg == "--samples") {
            options.samples = parseSampleCount(requireValue(i, arg), arg);
        } else if (arg == "--delta") {
            options.delta = std::stod(requireValue(i, arg));
        } else if (arg == "--seed") {
            options.seed = std::stoull(requireValue(i, arg));
        } else if (arg == "--full-coverage") {
            options.fullCoverage = true;
        } else if (arg == "--epsilon") {
            options.epsilon = std::stod(requireValue(i, arg));
        } else {
            throw std::runtime_error("Unrecognized option '" + arg + "'. Run --help for usage.");
        }
    }

    if (options.inputFile.empty()) {
        throw std::runtime_error("Missing required option --input.");
    }
    if (options.outputFile.empty()) {
        throw std::runtime_error("Missing required option --output.");
    }
    if (options.mode.empty()) {
        throw std::runtime_error("Missing required option --mode.");
    }
    if (options.mode != "learn-interval" && options.mode != "widen-interval" && options.mode != "sample-distribution") {
        throw std::runtime_error("Unknown mode '" + options.mode + "'. Supported modes: learn-interval, widen-interval, sample-distribution.");
    }
    if (options.mode == "widen-interval") {
        if (!options.delta.has_value()) {
            throw std::runtime_error("Mode 'widen-interval' requires --delta.");
        }
        if (options.samples.has_value()) {
            throw std::runtime_error("Mode 'widen-interval' does not use --samples.");
        }
        if (options.fullCoverage) {
            throw std::runtime_error("Mode 'widen-interval' does not sample the model, so --full-coverage has no effect.");
        }
    } else {
        if (options.samples.has_value() && options.delta.has_value()) {
            throw std::runtime_error("Cannot set both --samples and --delta.");
        }
        if (!options.samples.has_value() && !options.delta.has_value()) {
            throw std::runtime_error("Mode '" + options.mode + "' requires either --samples or --delta.");
        }
        if (options.epsilon.has_value()) {
            throw std::runtime_error("Mode '" + options.mode + "' does not support --epsilon; it is only supported for widen-interval.");
        }
    }
    if (options.epsilon.has_value() && options.epsilon.value() <= 0.0) {
        throw std::runtime_error("--epsilon must be positive.");
    }

    return options;
}

// Detects a model file's format from its extension: .umb, or .drn (optionally .drn.gz/.drn.xz). Storm's own
// detector throws an unhelpful message for a path with no extension at all, so that case is checked here first.
storm::io::ModelExportFormat detectFileFormat(std::string const& path) {
    if (std::filesystem::path(path).extension().empty()) {
        throw std::invalid_argument("File '" + path + "' has no extension. Supported extensions: .umb, .drn, .drn.gz, .drn.xz.");
    }
    return storm::io::getModelExportFormatFromFileExtension(path);
}

// Loads a model from `path`, dispatching on its file extension: .umb, or .drn (optionally .drn.gz/.drn.xz).
std::shared_ptr<storm::models::ModelBase> loadModel(std::string const& path) {
    switch (detectFileFormat(path)) {
        case storm::io::ModelExportFormat::Umb:
            return storm::api::buildExplicitUmbModel(path);
        case storm::io::ModelExportFormat::Drn:
            return storm::api::buildExplicitDRNModel(path);
        default:
            throw std::invalid_argument("Unsupported input file '" + path + "'. Supported extensions: .umb, .drn, .drn.gz, .drn.xz.");
    }
}

// Writes `model` to `path`, dispatching on its file extension the same way loadModel does.
template<typename ExportValueType>
void exportModel(std::shared_ptr<storm::models::sparse::Model<ExportValueType>> const& model, std::string const& path) {
    switch (detectFileFormat(path)) {
        case storm::io::ModelExportFormat::Umb:
            storm::api::exportSparseModelAsUmb(model, path);
            return;
        case storm::io::ModelExportFormat::Drn: {
            // The simple exportSparseModelAsDrn(model, path) overload defaults compression to None, which
            // would ignore a .drn.gz/.drn.xz extension; Default makes it resolved from that extension instead.
            storm::io::DirectEncodingExporterOptions drnOptions;
            drnOptions.compression = storm::io::CompressionMode::Default;
            storm::api::exportSparseModelAsDrn(model, path, drnOptions);
            return;
        }
        default:
            throw std::invalid_argument("Unsupported output file '" + path + "'. Supported extensions: .umb, .drn, .drn.gz, .drn.xz.");
    }
}

// Resolves the seed to use: the given --seed if any, otherwise a randomly drawn one (printed so the run
// can be reproduced later). Only called for modes that actually sample the model.
uint64_t resolveSeed(Options const& options) {
    uint64_t const seed = options.seed.value_or(std::random_device{}());
    if (!options.seed.has_value()) {
        std::cout << "Pass --seed " << seed << " to reproduce this run).\n";
    }
    return seed;
}

template<typename ValueType>
void runMode(storm::models::sparse::Model<ValueType> const& model, Options const& options) {
    if (options.mode == "learn-interval") {
        uint64_t const seed = resolveSeed(options);
        auto learnedModel =
            options.samples.has_value()
                ? ruffle::learnIMDPFromMDPByClopperPearsonUntilMaxSamples<ValueType>(model, options.lambda, options.samples.value(), seed, options.fullCoverage)
                : ruffle::learnIMDPFromMDPByClopperPearsonUntilL1Width<ValueType>(model, options.lambda, options.delta.value(), seed, options.fullCoverage);
        exportModel(learnedModel, options.outputFile);
    } else if (options.mode == "widen-interval") {
        std::optional<ValueType> const epsilon = options.epsilon;
        auto widenedModel = ruffle::widenModelIntervals<ValueType>(model, options.delta.value(), epsilon);
        exportModel(widenedModel, options.outputFile);
    } else {
        uint64_t const seed = resolveSeed(options);
        auto sampledModel = options.samples.has_value()
                                ? ruffle::sampleModelDistributionUntilMaxSamples<ValueType>(model, options.samples.value(), seed, options.fullCoverage)
                                : ruffle::sampleModelDistributionUntilL1Distance<ValueType>(model, options.delta.value(), seed, options.fullCoverage);
        exportModel(sampledModel, options.outputFile);
    }

    std::cout << "Wrote resulting model to " << options.outputFile << ".\n";
}

template<typename ValueType>
void processModel(std::shared_ptr<storm::models::sparse::Model<ValueType>>&& model, Options const& options) {
    if (model == nullptr) {
        throw std::runtime_error("Model is unsupported.");
    }
    runMode(*model, options);
}

}  // namespace

int main(int argc, char* argv[]) {
    std::optional<Options> options;
    try {
        options = parseOptions(argc, argv);
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    if (!options.has_value()) {
        // --help was given.
        return 0;
    }
    RuntimeReporter const runtimeReporter;

    storm::utility::setUp();
    storm::settings::initializeAll("ruffle", "ruffle");

    try {
        auto model = loadModel(options->inputFile);
        if (model->isExact()) {
            processModel(model->template as<storm::models::sparse::Model<storm::RationalNumber>>(), options.value());
        } else {
            processModel(model->template as<storm::models::sparse::Model<double>>(), options.value());
        }
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
