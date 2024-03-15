#################################################################################
# WaterTAP Copyright (c) 2020-2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################

# imports
from pyomo.environ import (
    ConcreteModel,
    value,
    TransformationFactory,
    units as pyunits,
    assert_optimal_termination,
    Block,
    Constraint,
    Objective,
)
from pyomo.network import Arc
from idaes.core import FlowsheetBlock
from idaes.core.solvers import get_solver
from idaes.models.unit_models.translator import Translator
from idaes.core.util.initialization import propagate_state
from idaes.models.unit_models import Product, Feed
from idaes.core import UnitModelCostingBlock
import idaes.core.util.scaling as iscale
from pyomo.util.check_units import assert_units_consistent
from watertap.unit_models.nanofiltration_ZO import NanofiltrationZO
from watertap.unit_models.pressure_changer import Pump
from watertap.costing import WaterTAPCosting
from watertap.core.zero_order_costing import ZeroOrderCosting
from watertap.core.wt_database import Database
import watertap.examples.flowsheets.full_treatment_train.model_components.seawater_ion_prop_pack_dhe as props
from idaes.core.util.scaling import (
    calculate_scaling_factors,
    constraint_scaling_transform,
    unscaled_variables_generator,
    unscaled_constraints_generator,
    badly_scaled_var_generator,
)
# get solver
solver = get_solver()

# setup flowsheet
m = ConcreteModel()
m.fs = FlowsheetBlock(dynamic=False)
m.fs.properties = props.PropParameterBlock()
m.fs.costing = WaterTAPCosting()

# create units
m.fs.feed = Feed(property_package=m.fs.properties)
m.fs.product = Product(property_package=m.fs.properties)
m.fs.disposal = Product(property_package=m.fs.properties)
m.fs.unit = NanofiltrationZO(property_package=m.fs.properties)
m.fs.P1 = Pump(property_package=m.fs.properties)

# connections
m.fs.s01 = Arc(source=m.fs.feed.outlet, destination=m.fs.P1.inlet)
m.fs.s02 = Arc(source=m.fs.P1.outlet, destination=m.fs.unit.inlet)
m.fs.s03 = Arc(source=m.fs.unit.permeate, destination=m.fs.product.inlet)
m.fs.s04 = Arc(source=m.fs.unit.retentate, destination=m.fs.disposal.inlet)

TransformationFactory("network.expand_arcs").apply_to(m)

# specify flowsheet
m.fs.feed.properties[0].pressure.fix(101325)  # feed pressure [Pa]
m.fs.feed.properties[0].temperature.fix(273.15 + 25)  # feed temperature [K]
# properties (cannot be fixed for initialization routines, must calculate the state variables)
m.fs.feed.properties[0].mass_frac_phase_comp["Liq", "Ca"] = 0.000891
m.fs.feed.properties[0].mass_frac_phase_comp["Liq", "Mg"] = 0.002878
m.fs.feed.properties[0].mass_frac_phase_comp["Liq", "SO4"] = 0.006745
m.fs.feed.properties[0].mass_frac_phase_comp["Liq", "Cl"] = 0.04366
m.fs.feed.properties[0].mass_frac_phase_comp["Liq", "Na"] = 0.02465
m.fs.feed.properties[0].mass_frac_phase_comp["Liq", "K"] = 0.00088799
m.fs.feed.properties[0].mass_frac_phase_comp["Liq", "CO3"] = 3.99997E-07
m.fs.feed.properties[0].mass_frac_phase_comp["Liq", "HCO3"] = 0.0003153

m.fs.feed.properties.calculate_state(
    var_args={
        ("flow_mass_phase_comp", ("Liq", "H2O")): 436.346,  # feed mass flow rate [kg/s]
        ("mass_frac_phase_comp", ("Liq", "Ca")): 0.000891,
        ("mass_frac_phase_comp", ("Liq", "Mg")): 0.002878,
        ("mass_frac_phase_comp", ("Liq", "SO4")): 0.006745,
        ("mass_frac_phase_comp", ("Liq", "Cl")): 0.04366,
        ("mass_frac_phase_comp", ("Liq", "Na")): 0.02465,
        ("mass_frac_phase_comp", ("Liq", "K")): 0.00088799,
        ("mass_frac_phase_comp", ("Liq", "CO3")): 3.99997E-07,
        ("mass_frac_phase_comp", ("Liq", "HCO3")): 0.0003153,
    },  # feed mass fractions [-]
    hold_state=True,  # fixes the calculated component mass flow rates
)
m.fs.P1.efficiency_pump.fix(0.80)  # pump efficiency [-]
m.fs.P1.outlet.pressure[0].fix(10e5)

# fully specify system
m.fs.unit.properties_permeate[0].pressure.fix(101325)
m.fs.unit.recovery_vol_phase.fix(0.6)
m.fs.unit.rejection_phase_comp[0, "Liq", "Na"].fix(0.01)
m.fs.unit.rejection_phase_comp[0, "Liq", "Ca"].fix(0.79)
m.fs.unit.rejection_phase_comp[0, "Liq", "Mg"].fix(0.94)
m.fs.unit.rejection_phase_comp[0, "Liq", "SO4"].fix(0.87)
m.fs.unit.rejection_phase_comp[0, "Liq", "K"].fix(0.01)
m.fs.unit.rejection_phase_comp[0, "Liq", "CO3"].fix(0.87) #similar to sulfate
m.fs.unit.rejection_phase_comp[0, "Liq", "HCO3"].fix(0.75) #https://doi.org/10.1080/19443994.2015.1135825
m.fs.unit.rejection_phase_comp[
    0, "Liq", "Cl"
] = 0.15  # guess, but electroneutrality enforced below
charge_comp = {"Na": 1, "Ca": 2, "Mg": 2, "SO4": -2, "Cl": -1, "K": 1,
               "CO3": -2, "HCO3": -1,}
m.fs.unit.eq_electroneutrality = Constraint(
    expr=0
    == sum(
        charge_comp[j]
        * m.fs.unit.feed_side.properties_out[0].conc_mol_phase_comp["Liq", j]
        for j in charge_comp
    )
)
constraint_scaling_transform(m.fs.unit.eq_electroneutrality, 1)

# scaling
m.fs.properties.set_default_scaling(
    "flow_mass_phase_comp", 1e-2, index=("Liq", "H2O")
)
m.fs.properties.set_default_scaling(
    "flow_mass_phase_comp", 1e-1, index=("Liq", "Na")
)
m.fs.properties.set_default_scaling(
    "flow_mass_phase_comp", 1e1, index=("Liq", "Ca")
)
m.fs.properties.set_default_scaling(
    "flow_mass_phase_comp", 1, index=("Liq", "Mg")
)
m.fs.properties.set_default_scaling(
    "flow_mass_phase_comp", 1, index=("Liq", "SO4")
)
m.fs.properties.set_default_scaling(
    "flow_mass_phase_comp", 1e-1, index=("Liq", "Cl")
)
m.fs.properties.set_default_scaling(
    "flow_mass_phase_comp", 1e1, index=("Liq", "K")
)
m.fs.properties.set_default_scaling(
    "flow_mass_phase_comp", 1e4, index=("Liq", "CO3")
)
m.fs.properties.set_default_scaling(
    "flow_mass_phase_comp", 1e1, index=("Liq", "HCO3")
)

iscale.set_scaling_factor(m.fs.P1.control_volume.work, 1e-3)

iscale.calculate_scaling_factors(m)

# initialize
m.fs.feed.initialize()
propagate_state(m.fs.s01)
m.fs.P1.initialize()
propagate_state(m.fs.s02)
m.fs.unit.initialize()
propagate_state(m.fs.s03)
m.fs.product.initialize()
propagate_state(m.fs.s04)
m.fs.disposal.initialize()

# solve model
results = solver.solve(m, tee=True)

# costing
m.fs.unit.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
m.fs.P1.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
m.fs.costing.cost_process()
m.fs.costing.add_annual_water_production(m.fs.product.properties[0].flow_vol)
m.fs.costing.add_LCOW(m.fs.product.properties[0].flow_vol)
m.fs.costing.add_specific_energy_consumption(m.fs.product.properties[0].flow_vol)
m.fs.costing.add_specific_electrical_carbon_intensity(
    m.fs.feed.properties[0].flow_vol
)
m.fs.costing.initialize()
# consistent units
assert_units_consistent(m)

# re-solve model
results = solver.solve(m, tee=True) 
assert_optimal_termination(results)

#print(results)
m.fs.feed.report()
m.fs.P1.report()
m.fs.unit.report()
m.fs.product.report()
m.fs.disposal.report()
m.fs.costing.total_capital_cost.display()
m.fs.costing.total_operating_cost.display()
m.fs.costing.LCOW.display()
m.fs.costing.specific_energy_consumption.display()
print("Permeate Flow (m3/s): " + "{:.4f}".format(value(m.fs.product.properties[0].flow_vol)))
print("Brine Flow (m3/s): " + "{:.4f}".format(value(m.fs.unit.feed_side.properties_out[0].flow_vol)))
