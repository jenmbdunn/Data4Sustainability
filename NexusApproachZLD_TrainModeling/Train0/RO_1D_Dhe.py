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
    Objective,
)
from pyomo.network import Arc
from idaes.core import FlowsheetBlock
from idaes.core.solvers import get_solver
from idaes.core.util.initialization import propagate_state
from idaes.models.unit_models import Product, Feed
from idaes.core import UnitModelCostingBlock
import idaes.core.util.scaling as iscale
from pyomo.util.check_units import assert_units_consistent
from watertap.unit_models.reverse_osmosis_1D import (
    ReverseOsmosis1D,
    ConcentrationPolarizationType,
    MassTransferCoefficient,
    PressureChangeType,
)
from watertap.unit_models.pressure_changer import Pump
from watertap.costing import WaterTAPCosting
from watertap.core.wt_database import Database
import watertap.property_models.seawater_prop_pack as prop_SW

# get solver
solver = get_solver()

# setup flowsheet
m = ConcreteModel()
m.fs = FlowsheetBlock(dynamic=False)
m.fs.prop_desal = prop_SW.SeawaterParameterBlock()
m.db = Database()


# create units
m.fs.feed = Feed(property_package=m.fs.prop_desal)
m.fs.P1 = Pump(property_package=m.fs.prop_desal)
m.fs.RO = ReverseOsmosis1D(
    property_package=m.fs.prop_desal,
    has_pressure_change=False,
    pressure_change_type=PressureChangeType.fixed_per_stage,
    mass_transfer_coefficient=MassTransferCoefficient.none,
    concentration_polarization_type=ConcentrationPolarizationType.none,
)

# connections
m.fs.s01 = Arc(source=m.fs.feed.outlet, destination=m.fs.P1.inlet)
m.fs.s02 = Arc(source=m.fs.P1.outlet, destination=m.fs.RO.inlet)

TransformationFactory("network.expand_arcs").apply_to(m)

# specify flowsheet
m.fs.feed.properties[0].pressure.fix(101325)  # feed pressure [Pa]
m.fs.feed.properties[0].temperature.fix(273.15 + 25)  # feed temperature [K]
# properties (cannot be fixed for initialization routines, must calculate the state variables)

m.fs.feed.properties[0].mass_frac_phase_comp["Liq", "TDS"] = 0.08
m.fs.feed.properties.calculate_state(
    var_args={
        ("flow_mass_phase_comp", ("Liq", "H2O")): 436.35,  # feed mass flow rate [kg/s]
        ("mass_frac_phase_comp", ("Liq", "TDS")): value(
            m.fs.feed.properties[0].mass_frac_phase_comp["Liq", "TDS"])
    },  # feed TDS mass fraction [-]
    hold_state=True,  # fixes the calculated component mass flow rates
)
m.fs.P1.efficiency_pump.fix(0.80)  # pump efficiency [-]
m.fs.P1.outlet.pressure[0].fix(70e5)
membrane_area = 11100 #membrane area = 50 * feed flow mass(kg/s) according to NF Test
A = 4.2e-12
B = 3.5e-8
pressure_atmospheric = 101325
m.fs.RO.area.fix(membrane_area)
m.fs.RO.A_comp.fix(A)
m.fs.RO.B_comp.fix(B)
m.fs.RO.permeate.pressure[0].fix(pressure_atmospheric)
m.fs.RO.length.fix(16)

# scaling
m.fs.prop_desal.set_default_scaling("flow_mass_phase_comp", 1e-3, index=("Liq", "H2O"))
m.fs.prop_desal.set_default_scaling(
    "flow_mass_phase_comp", 1e-2, index=("Liq", "TDS")
)
iscale.set_scaling_factor(m.fs.P1.control_volume.work, 1e-3)
iscale.set_scaling_factor(m.fs.RO.area, 1e-5)

iscale.calculate_scaling_factors(m)

# initialize
m.fs.feed.initialize()
propagate_state(m.fs.s01)
m.fs.P1.initialize()
propagate_state(m.fs.s02)
m.fs.RO.initialize()


# solve model
results = solver.solve(m, tee=True)
#Start optimizing
m.fs.RO.area.unfix()                  # membrane area (m^2)
m.fs.P1.outlet.pressure[0].unfix()     # feed pressure (Pa)
m.fs.RO.length.unfix()
m.fs.RO.area.setlb(1)
m.fs.RO.area.setub(None)
m.fs.P1.outlet.pressure[0].setlb(1e5)
m.fs.P1.outlet.pressure[0].setub(None)
m.fs.RO.recovery_vol_phase[0,'Liq'].fix(0.2) #vary this number to explore different recoveries

# costing
m.fs.costing = WaterTAPCosting()

m.fs.P1.costing = UnitModelCostingBlock(
    flowsheet_costing_block=m.fs.costing
)
m.fs.RO.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
m.fs.costing.cost_flow(m.fs.P1.work_mechanical[0], "electricity")
m.fs.costing.cost_process()
m.fs.costing.add_LCOW(m.fs.RO.mixed_permeate[0].flow_vol)
m.fs.costing.add_specific_energy_consumption(m.fs.RO.mixed_permeate[0].flow_vol)
m.fs.costing.initialize()


# consistent units
assert_units_consistent(m)

# optimize
m.fs.objective = Objective(expr=m.fs.costing.LCOW)
optimization_results = solver.solve(m)
assert_optimal_termination(results)

#print
m.fs.feed.report()
m.fs.P1.report()
m.fs.RO.report()
m.fs.costing.total_capital_cost.display()
m.fs.costing.total_operating_cost.display()
m.fs.costing.LCOW.display()
m.fs.costing.specific_energy_consumption.display()

print("Permeate flow (m3/s): " + "{:.4f}".format(value(m.fs.RO.mixed_permeate[0].flow_vol)))
print("Brine flow (m3/s): " + "{:.4f}".format(value(m.fs.RO.feed_side.properties[0, 1].flow_vol)))

if value(m.fs.P1.outlet.pressure[0]) >= 85e5:
    print("INFEASIBLE") #not feasible to operate conventional RO membranes above this pressure
