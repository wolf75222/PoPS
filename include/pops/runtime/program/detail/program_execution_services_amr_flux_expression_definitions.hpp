
struct FluxExpressionTerm {
  std::shared_ptr<const FluxBasis> basis;
  ExactPolynomial coefficient;
};

using FluxExpression = std::map<std::uint64_t, FluxExpressionTerm>;
using FluxExpressionRegistry = std::map<const field_type*, FluxExpression>;
using FluxExpressionUpdate = std::optional<FluxExpressionRegistry>;
