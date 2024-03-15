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

from pyomo.environ import ConcreteModel, assert_optimal_termination
from pyomo.util.check_units import assert_units_consistent
from idaes.core import FlowsheetBlock
from idaes.core.solvers import get_solver
from idaes.core import UnitModelCostingBlock
from pyomo.environ import TransformationFactory
from pyomo.network import Arc
from idaes.core.util.initialization import propagate_state
import idaes.core.util.scaling as iscale
from idaes.models.unit_models import Product, Feed
from watertap.unit_models.zero_order import (
    ChemicalAdditionZO,
    )
from watertap.core.wt_database import Database
from watertap.core.zero_order_properties import WaterParameterBlock
from watertap.core.zero_order_costing import ZeroOrderCosting

# get solver
solver = get_solver()

# setup model
m = ConcreteModel()
m.db = Database()
m.fs = FlowsheetBlock(dynamic=False)
m.fs.properties = WaterParameterBlock(solute_list=["chloride", "sodium", "magnesium",
                                                   "sulfate", "calcium", "potassium",
                                                   "carbonate", "bicarbonate"])
m.fs.costing = ZeroOrderCosting()

# create units
m.fs.feed = Feed(property_package=m.fs.properties)
m.fs.product = Product(property_package=m.fs.properties)
m.fs.antiscale = ChemicalAdditionZO(
    property_package=m.fs.properties, database=m.db, process_subtype="anti-scalant"
)
m.fs.hcl = ChemicalAdditionZO(
    property_package=m.fs.properties, database=m.db, process_subtype="hydrochloric_acid"
)


# connections
m.fs.s1 = Arc(source=m.fs.feed.outlet, destination=m.fs.antiscale.inlet)
m.fs.s2 = Arc(source=m.fs.antiscale.outlet, destination=m.fs.hcl.inlet)
m.fs.s3 = Arc(source=m.fs.hcl.outlet, destination=m.fs.product.inlet)

TransformationFactory("network.expand_arcs").apply_to(m)

# specify flowsheet        
m.fs.feed.properties[0].flow_mass_comp["H2O"].fix(436.3464)
m.fs.feed.properties[0].flow_mass_comp["chloride"].fix(20.7086)
m.fs.feed.properties[0].flow_mass_comp["sodium"].fix(11.6911)
m.fs.feed.properties[0].flow_mass_comp["magnesium"].fix(1.3649)
m.fs.feed.properties[0].flow_mass_comp["sulfate"].fix(3.1992)
m.fs.feed.properties[0].flow_mass_comp["calcium"].fix(0.4227)
m.fs.feed.properties[0].flow_mass_comp["potassium"].fix(0.4212)
m.fs.feed.properties[0].flow_mass_comp["carbonate"].fix(0.00018972)
m.fs.feed.properties[0].flow_mass_comp["bicarbonate"].fix(0.1495)
m.fs.antiscale.load_parameters_from_database(use_default_removal=True)
m.fs.hcl.load_parameters_from_database(use_default_removal=True)

# scaling
iscale.calculate_scaling_factors(m.fs)

# solve feed
m.fs.feed.initialize()

# initialize units
propagate_state(m.fs.s1)
m.fs.antiscale.initialize()
propagate_state(m.fs.s2)
m.fs.hcl.initialize()
propagate_state(m.fs.s3)

# solve model
results = solver.solve(m, tee=True)

# costing
m.fs.antiscale.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
m.fs.hcl.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
m.fs.costing.cost_process()
m.fs.costing.add_electricity_intensity(m.fs.product.properties[0].flow_vol)
m.fs.costing.add_LCOW(m.fs.product.properties[0].flow_vol)
m.fs.costing.initialize()

# consistent units
assert_units_consistent(m)

# re-solve model
results = solver.solve(m, tee=True)
assert_optimal_termination(results)

#print(results)
m.fs.feed.report()
m.fs.antiscale.report()
m.fs.hcl.report()
m.fs.product.report()
m.fs.costing.LCOW.display()
m.fs.costing.electricity_intensity.display()
m.fs.costing.total_capital_cost.display()
m.fs.costing.total_operating_cost.display()



