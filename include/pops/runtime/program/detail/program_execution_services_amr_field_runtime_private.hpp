private:
enum class ScratchKind : std::uint8_t { Rhs = 0, State = 1, Scalar = 2 };
/// Scratch identity carries the exact runtime owner inherited from the prototype.  This prevents
/// equal-layout multi-block candidates from crossing a generated Program route.
using ScratchKey = std::tuple<ScratchKind, int, int, std::int64_t, int>;
