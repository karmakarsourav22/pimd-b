#include "observable.h"
#include "simulation.h"
#include "units.h"

#include <ranges>
#include "mpi.h"
#include "winding.h"
#include <complex>

/**
 * @brief Generic observable class constructor
 */
Observable::Observable(const Simulation& _sim, int _freq, const std::string& _out_unit) :
    sim(_sim), freq(_freq), out_unit(_out_unit) {}

/**
 * Initializes observables with the given labels.
 * 
 * @param labels Labels of the quantities to be calculated
 */
void Observable::initialize(const std::vector<std::string>& labels) {
    for (const std::string& label : labels) {
        quantities.insert({ label, 0.0 });
    }
}

/**
 * @brief Resets the values of the observable to zero.
 * Useful for clearing results from the previous moleclar dynamics step.
 */
void Observable::resetValues() {
    for (auto it = quantities.begin(); it != quantities.end(); ++it) {
        it.value() = 0.0;
    }
}

/**
 * @brief Creates an observable object based on the observable type.
 *
 * @param observable_type Type of observable to create
 * @param _sim Simulation object
 * @param _freq Frequency at which the observable is calculated
 * @param _out_unit Output unit of the observable
 * @return std::unique_ptr<Observable> Pointer to the created observable object
 * @throw std::invalid_argument If the observable type is unknown
 */
std::unique_ptr<Observable> ObservableFactory::createQuantity(const std::string& observable_type,
                                                              const Simulation& _sim, int _freq,
                                                              const std::string& _out_unit) {
    if (observable_type == "energy") {
        return std::make_unique<EnergyObservable>(_sim, _freq, _out_unit);
    } else if (observable_type == "classical") {
        return std::make_unique<ClassicalObservable>(_sim, _freq, _out_unit);
    } else if (observable_type == "bosonic") {
        return std::make_unique<BosonicObservable>(_sim, _freq, _out_unit);
    } else if (observable_type == "winding") {
        return std::make_unique<WindingObservable>(_sim, _freq, _out_unit);
    } else if (observable_type == "rhok") {
        return std::make_unique<DensityKObservable>(_sim, _freq, _out_unit);
    } else if (observable_type == "rhok_grid") {
        return std::make_unique<DensityKObservableGrid>(_sim, _freq, _out_unit);
    } else if (observable_type == "rhok_SelnK") {
        return std::make_unique<DensityKObservableSelnK>(_sim, _freq, _out_unit);
    } else {
        throw std::invalid_argument("Unknown observable type.");
    }
}

// Constructor opens the file and writes the header
ObservablesLogger::ObservablesLogger(const std::string& filename, int _bead, const std::vector<std::unique_ptr<Observable>>& _observables)
    : bead(_bead), observables(_observables) {

    if (bead == 0) {
        file.open(std::format("{}/{}", Output::FOLDER_NAME, filename), std::ios::out | std::ios::app);

        if (!file.is_open()) {
            throw std::ios_base::failure(std::format("Failed to open {}.", Output::MAIN_FILENAME));
        }

        // Write the header
        file << std::format("{:^16s}", "step");

        for (const auto& observable : observables) {
            for (const auto& key : observable->quantities | std::views::keys) {
                file << std::vformat(" {:^16s}", std::make_format_args(key));
            }
        }

        file << '\n';
    }    
}

// Destructor closes the file if open
ObservablesLogger::~ObservablesLogger() {
    if (bead == 0 && file.is_open()) {
        file.close();
    }
}

// Log observables data to the file
void ObservablesLogger::log(int step) {
    if (bead == 0) {
        file << std::format("{:^16.8e}", static_cast<double>(step));
    }

    for (const auto& observable : observables) {
        // The inner loop is necessary because some observable classes can calculate
        // more than one observable (e.g., "energy" calculates both the kinetic and potential energies).
        for (const double& val : observable->quantities | std::views::values) {
            double quantity_value = 0.0;
            double local_quantity_value = val;

            // Sum the results from all processes (beads)
            MPI_Allreduce(&local_quantity_value, &quantity_value, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            if (bead == 0) {
                file << std::format(" {:^16.8e}", quantity_value);
            }
        }
    }

    if (bead == 0) {
        file << '\n';
    }
}

/**
 * @brief Energy observable class constructor.
 */
EnergyObservable::EnergyObservable(const Simulation& _sim, int _freq, const std::string& _out_unit) :
    Observable(_sim, _freq, _out_unit) {
    if (sim.external_potential_name == "free" && sim.interaction_potential_name == "free") {
        initialize({ "kinetic" });
    } else if (sim.external_potential_name == "free" || sim.interaction_potential_name == "free") {
        initialize({ "kinetic", "potential", "virial" });
    } else {
        initialize({ "kinetic", "potential", "ext_pot", "int_pot", "virial" });
    }
}

void EnergyObservable::calculate() {
    calculateKinetic();
    calculatePotential();
}

/**
 * @brief Calculates the quantum kinetic energy of the system using the primitive kinetic energy estimator.
 * Works both for distinguishable particles and bosons.
 */
void EnergyObservable::calculateKinetic() {
    // First, add the constant factor of d*N*P/(2*beta) to the kinetic energy (per bead)
    quantities["kinetic"] = 0.5 * NDIM * sim.natoms / sim.beta;

    // Then, subtract the spring energies. In the case of bosons, the exterior
    // spring energy requires separate treatment.
    if (sim.this_bead == 0 && sim.bosonic) {
        quantities["kinetic"] += sim.bosonic_exchange->primEstimator();
    } else {
        double spring_energy = sim.classicalSpringEnergy();
#if IPI_CONVENTION
        spring_energy /= sim.nbeads;
#endif

        quantities["kinetic"] -= spring_energy;
    }

    quantities["kinetic"] = Units::convertToUser("energy", out_unit, quantities["kinetic"]);
}

/**
 * @brief Calculates the quantum potential energy of the system, based on the potential energy estimator.
 * The potential energy is the sum of the external potential energy and the interaction potential energy 
 * across all time-slices, divided by the number of beads. In addition, the method calculates the virial
 * kinetic energy of the system.
 */
void EnergyObservable::calculatePotential() {
    double potential = 0.0;  // Total potential energy
    double virial = 0.0;     // Virial kinetic energy
    double int_pot = 0.0;    // Potential energy due to interactions
    double ext_pot = 0.0;    // Potential energy due to external field

    if (sim.external_potential_name != "free") {
        ext_pot = sim.ext_potential->V(sim.coord);
        potential += ext_pot;

        dVec physical_forces(sim.natoms);
        physical_forces = (-1.0) * sim.ext_potential->gradV(sim.coord);

        for (int ptcl_idx = 0; ptcl_idx < sim.natoms; ++ptcl_idx) {
            for (int axis = 0; axis < NDIM; ++axis) {
                virial -= sim.coord(ptcl_idx, axis) * physical_forces(ptcl_idx, axis);
            }
        }
    }

    if (sim.int_pot_cutoff != 0.0) {
        for (int ptcl_one = 0; ptcl_one < sim.natoms; ++ptcl_one) {
            for (int ptcl_two = ptcl_one + 1; ptcl_two < sim.natoms; ++ptcl_two) {
                dVec diff = sim.getSeparation(ptcl_one, ptcl_two, sim.apply_mic_potential);  // Vectorial distance

                if (const double distance = diff.norm(); distance < sim.int_pot_cutoff || sim.int_pot_cutoff < 0.0) {
                    dVec force_on_one = (-1.0) * sim.int_potential->gradV(diff);

                    double int_pot_val = sim.int_potential->V(diff);
                    potential += int_pot_val;
                    int_pot += int_pot_val;

                    for (int axis = 0; axis < NDIM; ++axis) {
                        virial -= sim.coord(ptcl_one, axis) * force_on_one(0, axis);
                    }
                }
            }
        }
    }

    if (sim.external_potential_name != "free" && sim.interaction_potential_name != "free") {
        ext_pot /= sim.nbeads;
        int_pot /= sim.nbeads;

        quantities["ext_pot"] = Units::convertToUser("energy", out_unit, ext_pot);
        quantities["int_pot"] = Units::convertToUser("energy", out_unit, int_pot);
    }

    if (sim.external_potential_name != "free" || sim.interaction_potential_name != "free") {
        potential /= sim.nbeads;
        virial *= 0.5 / sim.nbeads;

        quantities["potential"] = Units::convertToUser("energy", out_unit, potential);
        quantities["virial"] = Units::convertToUser("energy", out_unit, virial);
    }
}

/**
 * @brief Classical observable class constructor.
 */
ClassicalObservable::ClassicalObservable(const Simulation& _sim, int _freq, const std::string& _out_unit) :
    Observable(_sim, _freq, _out_unit) {
    initialize({ "temperature", "cl_kinetic", "cl_spring" });
}

void ClassicalObservable::calculate() {
    calculateKineticEnergy();
    calculateSpringEnergy();
}

/**
 * @brief Calculates the classical kinetic energy of the ring polymers, as well as the temperature of the system.
 */
void ClassicalObservable::calculateKineticEnergy() {
    double kinetic_energy = 0.0;

    for (int ptcl_idx = 0; ptcl_idx < sim.natoms; ++ptcl_idx) {
        for (int axis = 0; axis < NDIM; ++axis) {
            kinetic_energy += sim.momenta(ptcl_idx, axis) * sim.momenta(ptcl_idx, axis);
        }
    }

    kinetic_energy *= 0.5 / sim.mass;
    quantities["cl_kinetic"] = Units::convertToUser("energy", out_unit, kinetic_energy);

    // Temperature is calculated according to Tolman's equipartition theorem as the average kinetic 
    // energy per degree of freedom. This might not be accurate for systems with constraints.
    // See [J. Chem. Theory Comput. 2019, 15, 1, 84-94.] for a discussion on the topic.

    /// @todo When zeroing the center of mass motion, the number of degrees of freedom must be reduced by NDIM.
    double dof = NDIM * sim.natoms * sim.nbeads;
    double temperature = 2.0 * kinetic_energy / (dof * Constants::kB);

    // In the i-Pi convention, the ring-polymer simulation is performed at a temperature that is P times higher
    // than the actual (quantum) temperature. Therefore, to ensure the quantum temperature is calculated correctly,
    // one must divide the classical temperature by the number of beads.
#if IPI_CONVENTION
    temperature /= sim.nbeads;
#endif

    /// @todo Allow conversion to different temperature units
    quantities["temperature"] = Units::convertToUser("temperature", "kelvin", temperature);
}

/**
 * @brief Calculates the spring energy of the classical ring-polymer system.
 * In the bosonic case, one must use the effective bosonic spring potential
 * for the exterior connection.
 */
void ClassicalObservable::calculateSpringEnergy() {
    double spring_energy;

    if (sim.this_bead == 0 && sim.bosonic) {
        spring_energy = sim.bosonic_exchange->effectivePotential();
    } else {
        spring_energy = sim.classicalSpringEnergy();
    }

    quantities["cl_spring"] = Units::convertToUser("energy", out_unit, spring_energy);
}

/**
 * @brief Bosonic observable class constructor.
 */
BosonicObservable::BosonicObservable(const Simulation& _sim, int _freq, const std::string& _out_unit) :
    Observable(_sim, _freq, _out_unit) {
    initialize({ "prob_dist", "prob_all" });
}

/**
 * @brief Calculates quantities pertaining to bosonic exchange.
 */
void BosonicObservable::calculate() {
    if (sim.this_bead == 0 && sim.bosonic) {
        quantities["prob_dist"] = sim.bosonic_exchange->getDistinctProbability();
        quantities["prob_all"] = sim.bosonic_exchange->getLongestProbability();
    }
}

///**
// * @brief Winding observable class constructor.
// */
//WindingObservable::WindingObservable(const Simulation& _sim, int _freq, const std::string& _out_unit) :
//    Observable(_sim, _freq, _out_unit) {
//    // Maybe remove this later, because we might want to be able to compile
//    // with an arbitrary NDIM and then not use this observable (in which case there
//    // should be no error). The error should probably be thrown only when trying
//    // to use the observable with an unsupported NDIM.
//    static_assert(NDIM <= 3, "NDIM can be at most 3 (for x, y, z)");
//
//#if NDIM == 1
//    initialize({ "W, W2" });
//#elif NDIM == 2
//    initialize({ "W_x", "W_y", "W2" });
//#elif NDIM == 3
//        initialize({ "W_x", "W_y", "W_z", "W2" });
//#endif
//}
//
///**
// * @brief Calculates quantities pertaining to bosonic exchange.
// */
//void WindingObservable::calculate() {
//    int key_axis = 0;
//
//    // It is assumed that the iteration is performed in the correct order,
//    // which depends on the order of the keys in the quantities map.
//    for (const std::string& key : quantities | std::views::keys | std::views::take(NDIM)) {
//        quantities[key] = 0.0;
//
//        if (sim.this_bead == 0 && sim.bosonic) {
//            quantities[key] = sim.bosonic_exchange->windingEstimator(key_axis);
//        } else {
//            for (int ptcl_idx = 0; ptcl_idx < sim.natoms; ++ptcl_idx) {
//                if (sim.include_wind_corr) {
//                    double diff_next = sim.coord(ptcl_idx, key_axis) - sim.next_coord(ptcl_idx, key_axis);
//
//                    WindingProbability wind_prob(diff_next, sim.max_wind, sim.beta_half_k, sim.size);
//                    quantities[key] -= wind_prob.getExpectation();
//                }
//
//            }
//        }
//        ++key_axis;
//    }
//
//    if (!sim.include_wind_corr) {
//        // This whole estimator doesn't make sense if the winding correction is not included
//        // Perhaps throw an error here, or in the initialization of the observable?
//    }
//
//    // W2 doesn't work with bosons currently
//    quantities["W2"] = 0.0;
//
//    for (int ptcl_idx = 0; ptcl_idx < sim.natoms; ++ptcl_idx) {
//        for (int axis = 0; axis < NDIM; ++axis) {
//            double diff_next = sim.coord(ptcl_idx, axis) - sim.next_coord(ptcl_idx, axis);
//            int wind_mic = -static_cast<int>(std::floor(diff_next / sim.size + 0.5));
//            WindingProbability wind_prob(diff_next, sim.max_wind, sim.beta_half_k, sim.size);
//
//            for (int wind_idx = 1; wind_idx <= sim.max_wind; ++wind_idx) {
//                int wind2 = wind_idx * wind_idx;
//                quantities["W2"] += wind2 * wind_prob.getProbability(wind_idx);
//                quantities["W2"] += wind2 * wind_prob.getProbability(-wind_idx);
//            }
//
//            quantities["W2"] -= wind_mic * wind_mic * wind_prob.getProbability(wind_mic);
//        }
//    }
//}

/**
 * @brief Winding observable class constructor.
 */
WindingObservable::WindingObservable(const Simulation& _sim, int _freq, const std::string& _out_unit) :
    Observable(_sim, _freq, _out_unit) {
#if NDIM == 1
    initialize({ "Wx2" });
#elif NDIM == 2
    initialize({ "Wx2", "Wy2" });
#elif NDIM == 3
        initialize({ "Wx2", "Wy2", "Wz2" });
#endif
}

/**
 * @brief Calculates quantities pertaining to bosonic exchange.
 */
void WindingObservable::calculate() {
    int key_axis = 0;

    // It is assumed that the iteration is performed in the correct order,
    // which depends on the order of the keys in the quantities map.
    for (const std::string& key : quantities | std::views::keys | std::views::take(NDIM)) {
        quantities[key] = 0.0;

        if (sim.this_bead == 0 && sim.bosonic) {
            quantities[key] = sim.bosonic_exchange->windingEstimator(key_axis);
        } else {
            for (int ptcl_idx = 0; ptcl_idx < sim.natoms; ++ptcl_idx) {
                double diff_next = sim.coord(ptcl_idx, key_axis) - sim.next_coord(ptcl_idx, key_axis);

                WindingProbability wind_prob(diff_next, sim.max_wind, sim.beta_half_k, sim.size);
                quantities[key] += wind_prob.getSquaredExpectation();
            }
        }
        ++key_axis;
    }
}

/**
 * @brief Calculates rho(-k, t) => the Fourier transform of the density operator, rho(x) = \sum_i \delta(x - x_i)
 */
DensityKObservable::DensityKObservable(const Simulation& _sim, int _freq, const std::string& _out_unit) :
    Observable(_sim, _freq, _out_unit) {
    initialize({ "rhop125KF_re", "rhop125KF_im", "rhop25KF_re", "rhop25KF_im", "rhop5KF_re", "rhop5KF_im", "rhoKF_re", "rhoKF_im", "rho2KF_re", "rho2KF_im" });
}

void DensityKObservable::calculate() {
    calculateDensityKF();
}

void DensityKObservable::calculateDensityKF() {
    std::complex<double> rho_local_p125KF(0.0, 0.0);
    std::complex<double> rho_local_p25KF(0.0, 0.0);
    std::complex<double> rho_local_p5KF(0.0, 0.0);
    std::complex<double> rho_local_KF(0.0, 0.0);
    std::complex<double> rho_local_2KF(0.0, 0.0);

    // Local contribution from the current bead only
    for (int ptcl_idx = 0; ptcl_idx < sim.natoms; ++ptcl_idx) {
        const double x = sim.coord(ptcl_idx, 0);  // 1D case

	rho_local_p125KF += std::exp(std::complex<double>(0.0, 0.125 * sim.KF * x));
        rho_local_p25KF  += std::exp(std::complex<double>(0.0, 0.25 * sim.KF * x));
	rho_local_p5KF   += std::exp(std::complex<double>(0.0, 0.5 * sim.KF * x));
	rho_local_KF     += std::exp(std::complex<double>(0.0, 1. * sim.KF * x));
	rho_local_2KF    += std::exp(std::complex<double>(0.0, 2. * sim.KF * x));
    }

    quantities["rhop125KF_re"] = rho_local_p125KF.real() / sim.nbeads;
    quantities["rhop125KF_im"] = rho_local_p125KF.imag() / sim.nbeads;
    quantities["rhop25KF_re"]  = rho_local_p25KF.real() / sim.nbeads;
    quantities["rhop25KF_im"]  = rho_local_p25KF.imag() / sim.nbeads;
    quantities["rhop5KF_re"]   = rho_local_p5KF.real() / sim.nbeads;
    quantities["rhop5KF_im"]   = rho_local_p5KF.imag() / sim.nbeads;
    quantities["rhoKF_re"]     = rho_local_KF.real() / sim.nbeads;
    quantities["rhoKF_im"]     = rho_local_KF.imag() / sim.nbeads;
    quantities["rho2KF_re"]    = rho_local_2KF.real() / sim.nbeads;
    quantities["rho2KF_im"]    = rho_local_2KF.imag() / sim.nbeads;
}

/**
 * @brief Calculates rho(-k, t), the Fourier transform of the density operator
 * rho(x) = \sum_i \delta(x - x_i)
 */

DensityKObservableGrid::DensityKObservableGrid(const Simulation& _sim, int _freq, const std::string& _out_unit) :
    Observable(_sim, _freq, _out_unit)
{
    // Build multipliers here, where sim is accessible
    kf_multipliers.reserve(sim.nK);
    double step = 1.0 / static_cast<double>(sim.nK);
    for (int i = 1; i <= sim.nK; ++i)
        kf_multipliers.push_back(i * step);

    std::vector<std::string> headers;
    headers.reserve(2 * kf_multipliers.size());
    for (std::size_t i = 0; i < kf_multipliers.size(); ++i) {
        std::string key = "rho_nk" + std::to_string(i + 1);
        headers.push_back(key + "_re");
        headers.push_back(key + "_im");
    }
    initialize(headers);
}

void DensityKObservableGrid::calculate() {
    calculateDensityKFGrid();
}

void DensityKObservableGrid::calculateDensityKFGrid() {
    std::vector<std::complex<double>> rho_local(kf_multipliers.size(), {0.0, 0.0});

    for (int ptcl_idx = 0; ptcl_idx < sim.natoms; ++ptcl_idx) {
        const double x = sim.coord(ptcl_idx, 0);
        for (std::size_t i = 0; i < kf_multipliers.size(); ++i) {
            double k = kf_multipliers[i] * sim.KF;
            rho_local[i] += std::exp(std::complex<double>(0.0, k * x)); // ρ(-k)
        }
    }

    for (std::size_t i = 0; i < kf_multipliers.size(); ++i) {
        std::string key = "rho_nk" + std::to_string(i + 1);
        quantities[key + "_re"] = rho_local[i].real() / sim.nbeads; // normalize by N
        quantities[key + "_im"] = rho_local[i].imag() / sim.nbeads;
    }
}

/**
 * @brief Calculates rho(-k, t), the Fourier transform of the density operator
 * rho(x) = \sum_i \delta(x - x_i)
 */

DensityKObservableSelnK::DensityKObservableSelnK(const Simulation& _sim, int _freq, const std::string& _out_unit) :
    Observable(_sim, _freq, _out_unit)
{
    double inv_nK = 1.0 / static_cast<double>(sim.nK);

    // Generate 1, 2, 4, 8, …, nK  (nK is always a power of 2)
    k_indices.reserve(32); // at most log2(nK) + 1 entries
    for (int i = 1; i <= sim.nK; i <<= 1)
        k_indices.push_back(i);

    kf_multipliers.reserve(k_indices.size());
    for (int idx : k_indices)
        kf_multipliers.push_back(idx * inv_nK);

    std::vector<std::string> headers;
    headers.reserve(2 * k_indices.size());
    for (int idx : k_indices) {
        std::string key = "rho_nk" + std::to_string(idx);
        headers.push_back(key + "_re");
        headers.push_back(key + "_im");
    }
    initialize(headers);
}

void DensityKObservableSelnK::calculate() {
    calculateDensityKFSelnK();
}

void DensityKObservableSelnK::calculateDensityKFSelnK() {
    std::vector<std::complex<double>> rho_local(k_indices.size(), {0.0, 0.0});

    for (int ptcl_idx = 0; ptcl_idx < sim.natoms; ++ptcl_idx) {
        const double x = sim.coord(ptcl_idx, 0);
        for (std::size_t i = 0; i < k_indices.size(); ++i) {
            double k = kf_multipliers[i] * sim.KF;
            rho_local[i] += std::exp(std::complex<double>(0.0, k * x));
        }
    }

    for (std::size_t i = 0; i < k_indices.size(); ++i) {
        std::string key = "rho_nk" + std::to_string(k_indices[i]);
        quantities[key + "_re"] = rho_local[i].real() / sim.nbeads;
        quantities[key + "_im"] = rho_local[i].imag() / sim.nbeads;
    }
}


