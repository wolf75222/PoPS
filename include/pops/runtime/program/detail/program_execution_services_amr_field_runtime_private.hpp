private:
enum class ScratchKind : std::uint8_t { Rhs = 0, State = 1, Scalar = 2 };
struct PreparedScratchDescriptor {
  int runtime_block = -1;
  /// -1 owns one field per hierarchy level; otherwise the sole field is bound to this level.
  int declared_level = -1;
  int ncomp = 0;
  Extent<Dim> ghosts{};
};
using PreparedScratchStorage =
    std::vector<std::array<std::vector<std::optional<std::vector<field_type>>>, 3>>;
using PreparedScratchDescriptors =
    std::vector<std::array<std::vector<std::optional<PreparedScratchDescriptor>>, 3>>;
/// Scratch identity carries the exact runtime owner inherited from the prototype.  This prevents
/// equal-layout multi-block candidates from crossing a generated Program route.
using ScratchKey = std::tuple<ScratchKind, int, int, std::int64_t, int>;
