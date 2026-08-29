
struct FluxBasisFace {
  ::pops::amr::reflux::FaceLedgerRole role = ::pops::amr::reflux::FaceLedgerRole::Coarse;
  int axis = 0;
  Index<Dim> face{};
  Index<Dim> coarse_face{};
  double face_measure = 0.0;
  std::vector<Real> flux_density;
};

enum class FluxBasisProvider : std::uint8_t {
  PreparedResidual = 0,
  PreparedDefaultFlux = 1,
  ExactFace = 2,
  NamedCell = 3,
};

struct FluxBasis {
  std::uint64_t identity = 0;
  std::size_t runtime_block = 0;
  int level = 0;
  runtime::multiblock::BoundaryEvaluationPoint point{};
  int rhs_identity = -1;
  FluxBasisProvider provider = FluxBasisProvider::PreparedResidual;
  ::pops::amr::ClockWindow window{};
  std::vector<FluxBasisFace> faces;
};
