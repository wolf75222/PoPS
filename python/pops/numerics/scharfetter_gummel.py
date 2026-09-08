"""One fitted face construction consuming an exact drift/diffusion occurrence pair."""
import math
from pops.model.balance_analysis import fitted_balance_supported
from .diffusion import Diffusion


class ScharfetterGummel(Diffusion):
    category="fitted_drift_diffusion"
    native_id="pops::runtime::program::PreparedDiffusion::apply_fitted"

    def __init__(self,*,drift,flux):
        from pops.physics.drift_diffusion import DriftFluxHandle

        if type(drift) is not DriftFluxHandle:
            raise TypeError("ScharfetterGummel requires an exact physical drift flux")
        self.drift=drift
        super().__init__(flux=flux)

    def validate(self):
        from pops._ir.expr import Const
        from pops._ir.quantity import QuantityRef

        super().validate()
        if self.law.dimension!=1 or self.drift.state!=self.law.state:
            raise ValueError("fitted drift diffusion requires one exact Dim1 scalar state")
        density=self.law.variable
        if not isinstance(density,QuantityRef) or density.handle!=self.law.state or density.index!=0:
            raise ValueError("ScharfetterGummel requires diffusion of the conserved scalar density")
        diffusion=self.law.coefficients[0][0]
        mobility=self.drift.law.mobility
        if not isinstance(diffusion,Const) or not isinstance(mobility,Const):
            raise ValueError("selected fitted realization requires constant diffusivity and mobility")
        if not math.isfinite(diffusion.value) or diffusion.value<=0 or not math.isfinite(mobility.value) or mobility.value<0:
            raise ValueError("fitted coefficients require finite D>0 and mobility>=0")
        for density_bc,potential_bc in zip(self.law.boundaries,self.drift.law.boundaries,strict=True):
            if density_bc.kind!=potential_bc.kind or density_bc.kind=="conormal":
                raise ValueError("fitted face boundary requires paired periodic or value density/potential traces")
        return True

    def validate_rate_contract(self,contract):
        if contract["state"]!=self.law.state or contract.get("flux") not in (None,()):
            raise ValueError("fitted drift/diffusion replaces neither another transport flux nor another state")
        return True

    def validate_balance_view(self,view):
        if not fitted_balance_supported(view):
            raise ValueError("ScharfetterGummel must cover drift and diffusion together exactly once; split or repeated coverage is forbidden")
        for row in view.occurrences:
            if row.kind=="drift" and (row.payload!=self.drift or row.coefficient!=-1):
                raise ValueError("fitted construction requires exactly -div of its declared drift")
            if row.kind=="diffusion" and (row.payload!=self.flux or row.coefficient!=1):
                raise ValueError("fitted construction requires exactly +div of its declared diffusion")
        return self.validate()

    def resolve_references(self,resolver):
        result=super().resolve_references(resolver)
        result.drift=resolver(self.drift)
        return result

    def to_data(self):
        data=super().to_data()
        data.update(method="scharfetter_gummel",drift=self.drift.canonical_identity(),
                    drift_law=self.drift.law.to_data(),face_sampling="potential_difference",
                    coverage="one_joint_drift_diffusion_flux",bernoulli="stable_series_expm1_asymptotic")
        return data
