#pragma once

#include "common.h"

/* -------------- Basic potential class -------------- */
class Potential {
public:
    Potential();
    virtual ~Potential() = default;

    // Potential
    virtual double V(const dVec& x) {
        return 0.0;
    }

    // Potential gradient
    virtual dVec gradV(const dVec& x) {
        return dVec(x.len()); // Zero vector of the same length as x
    }

    // Potential laplacian
    virtual double laplacianV(const dVec& x) {
        return 0.0;
    }

    // Tail correction
    /// @todo Implement the tail correction
    double tailV;
};

/* -------------- Isotropic harmonic potential -------------- */
class HarmonicPotential : public Potential {
public:
    HarmonicPotential(double mass, double omega);
    ~HarmonicPotential() override = default;

    // Potential
    double V(const dVec& x) override;

    // Potential gradient
    dVec gradV(const dVec& x) override;

    // Potential laplacian
    double laplacianV(const dVec& x) override;

private:
    double mass;  // Mass of the particle experiencing the harmonic potential
    double omega; // Angular frequency of the oscillator
    double k;     // Force constant of the oscillator (k=mw^2)
};

/* -------------- Isotropic double well potential -------------- */
class DoubleWellPotential : public Potential {
public:
    DoubleWellPotential(double mass, double strength, double loc);
    ~DoubleWellPotential() override = default;

    // Potential
    double V(const dVec& x) override;

    // Potential gradient
    dVec gradV(const dVec& x) override;

    // Potential laplacian
    double laplacianV(const dVec& x) override;

private:
    double mass;      // Mass of the particle experiencing the double-well potential
    double strength;  // Height of the well
    double loc;       // Distance of the wells from the origin
};

/* -------------- Isotropic cosine potential -------------- */
class CosinePotential : public Potential {
public:
    CosinePotential(double amplitude, double wavelength, double phase);
    ~CosinePotential() override = default;

    // Potential
    double V(const dVec& x) override;

    // Potential gradient
    dVec gradV(const dVec& x) override;

    // Potential laplacian
    double laplacianV(const dVec& x) override;

private:
    double amplitude;   // Amplitude of the potential (V_0)
    double wavelength;  // Wavelength of the potential (L)
    double phase;       // Phase shift of the potential (phi)
    double k;           // Wavenumber of the potential (k = 2*pi/L)
};

/* -------------- Dipole potential -------------- */
class DipolePotential : public Potential {
public:
    DipolePotential(double strength);
    ~DipolePotential() override = default;

    // Potential
    double V(const dVec& x) override;

    // Potential gradient
    dVec gradV(const dVec& x) override;

    // Potential laplacian
    double laplacianV(const dVec& x) override;

private:
    double strength;
};

/* -------------- Aziz potential -------------- */
class AzizPotential : public Potential {
public:
    AzizPotential();
    ~AzizPotential() override = default;

    // Potential
    double V(const dVec& x) override;

    // Potential gradient
    dVec gradV(const dVec& x) override;

    // Potential laplacian
    double laplacianV(const dVec& x) override;

private:
    double rm, A, epsilon, alpha, D, C6, C8, C10;

    // The auxiliary F-function for the Aziz potential
    double F(const double x) const {
        return (x < D ? exp(-(D / x - 1.0) * (D / x - 1.0)) : 1.0);
    }

    // The derivative of the F-function
    double dF(const double x) const {
        double ix = 1.0 / x;
        double r = 2.0 * D * ix * ix * (D * ix - 1.0) * exp(-(D * ix - 1.0) * (D * ix - 1.0));
        return (x < D ? r : 0.0);
    }

    // The 2nd derivative of the F-function
    double d2F(const double x) const {
        double ix = 1.0 / x;
        double r = 2.0 * D * ix * ix * ix * (2.0 * D * D * D * ix * ix * ix - 4.0 * D * D * ix * ix
            - D * ix + 2.0) * exp(-(D * ix - 1.0) * (D * ix - 1.0));
        return (x < D ? r : 0.0);
    }
};

/* -------------- Repulsive gaussian approximation of delta potential -------------- */
class GaussianDeltaRepulsion : public Potential {
public:
    GaussianDeltaRepulsion(double g, double sigma);
    ~GaussianDeltaRepulsion() override = default;

    // Potential
    double V(const dVec& x) override;

    // Potential gradient
    dVec gradV(const dVec& x) override;

    // Potential laplacian
    double laplacianV(const dVec& x) override;

private:
    double g;   // V(r_ij) = (g / sqrt(2 * pi * sigma * sigma)) * exp(- 0.5 * (r_ij / sigma)^2)
    double sigma;
    double strength;   // strength = (g / sqrt(2 * pi * sigma * sigma))
};

/* -------------- Isotropic double well potential -------------- */
class QuarticPotential : public Potential {
public:
    QuarticPotential(double mass, double k, double c);
    ~QuarticPotential() override = default;

    // Potential
    double V(const dVec& x) override;

    // Potential gradient
    dVec gradV(const dVec& x) override;

    // Potential laplacian
    double laplacianV(const dVec& x) override;

private:
    double mass;      // Mass of the particle experiencing the quartic potential
    double k;         // k has the dimension of energy ... V(r) = 0.25 k (r/c)^4
    double c;         // It has the dimension of length.

    double c4;        // c4 = c * c * c * c
};

/* -------------- Anharmonic potential -------------- */
class AnharmonicAnisotropic : public Potential {
public:
    AnharmonicAnisotropic(double mass, double omega, double c3, double c4);
    ~AnharmonicAnisotropic() override = default;

    // Potential
    double V(const dVec& x) override;

    // Potential gradient
    dVec gradV(const dVec& x) override;

    // Potential laplacian
    double laplacianV(const dVec& x) override;

private:
    double mass;         // V(r) = 0.5 * mass * (omega)^2 *
    double omega;        //      sum_xi={x,y,z} ( xi**2 + c3 * xi**3 + c4 * xi**4 )
    double c3;           //
    double c4;

    double k;            // = mass * omega * omega
};

