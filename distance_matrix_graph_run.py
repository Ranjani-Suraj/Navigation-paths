# Main Utility Script

# Importing Scripts
import gmaps_api_ops
import BuildingGraph

# Google Maps Side of things
gmaps_api_ops.run(["Pittsburgh PA", "State College PA", "Erie PA", "Harrisburg PA", "Philadelphia PA"])

# Building Graph, after which, you can do anything you want!
graph = BuildingGraph.build_graph()
