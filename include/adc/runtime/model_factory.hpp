#pragma once

#include <adc/core/types.hpp>
#include <adc/model/charged_fluid.hpp>   // ChargedEuler, ChargedEulerIsothermal (+ Euler)
#include <adc/model/diocotron.hpp>
#include <adc/model/euler_poisson.hpp>

#include <stdexcept>
#include <string>

/// @file
/// @brief Fabrique partagee des modeles physiques de la composition runtime.
///
/// Centralise la LISTE des modeles connus des facades runtime (System, AmrSystem) en UN
/// seul endroit : ajouter un modele = une entree ici, les deux facades en heritent. Ce
/// n'est pas un plugin a l'execution (les modeles sont du code device compile), mais cela
/// supprime la duplication du dispatch par tag.

namespace adc::detail {

/// Champs de configuration lus pour construire un modele (communs a System / AmrSystem).
struct ModelParams {
  double B0, n_i0, alpha, gamma, cs2, four_pi_G, rho0, charge;
};

/// Construit le modele designe par @p tag et appelle `visitor(model)`.
///
/// @param tag     "diocotron" | "electron_euler" | "ion_isothermal" | "euler_poisson"
/// @param p        parametres du modele (charge incluse)
/// @param visitor  appele avec le modele concret (lambda generique `[](auto m){...}`)
/// @throws std::runtime_error si @p tag est inconnu.
template <class Visitor>
void dispatch_model(const std::string& tag, const ModelParams& p, Visitor&& visitor) {
  if (tag == "diocotron") {
    visitor(Diocotron{Real(p.B0), Real(p.n_i0), Real(p.alpha)});
  } else if (tag == "electron_euler") {
    visitor(ChargedEuler{Euler{Real(p.gamma)}, Real(p.charge), Real(p.charge)});
  } else if (tag == "ion_isothermal") {
    visitor(ChargedEulerIsothermal{Real(p.cs2), Real(p.charge), Real(p.charge)});
  } else if (tag == "euler_poisson") {
    EulerPoisson m;
    m.hydro.gamma = Real(p.gamma);
    m.four_pi_G = Real(p.four_pi_G);
    m.rho0 = Real(p.rho0);
    m.coupling_sign = Real(p.charge);  // +1 auto-gravite, -1 electrostatique (Langmuir)
    visitor(m);
  } else {
    throw std::runtime_error(
        "modele inconnu '" + tag +
        "' (diocotron|electron_euler|ion_isothermal|euler_poisson)");
  }
}

}  // namespace adc::detail
