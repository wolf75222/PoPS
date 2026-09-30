#include <pops/core/identity/physical_dimension_json.hpp>
#include <iostream>
// Actual reader; only its documented invalid_argument refusal is caught.
int main() {
  std::string line;
  while (std::getline(std::cin,line)) {
    try {
      pops::identity::require_canonical_physical_dimension_json(line);
      std::cout << "accept\n";
    } catch (const std::invalid_argument&) {
      std::cout << "refuse\n";
    }
  }
}
