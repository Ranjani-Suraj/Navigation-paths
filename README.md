# Tartanhacks2024

The main project repository for our team for TartanHacks 2024. The current running project idea is to address the PLS Challenge, by using some form of graph modification in tandem with machine learning to optimize routes in near real time.

Problem definition: Similar to an Amazon delivery truck, Semi-trucks need to deliver to a certain number of cities before RTB to restock. This would be easy to solve with the travelling salesman problem. We're going to try and solve this with both a classical and quantum approach, to show the increases in efficiency with the latter.

Factors:
- Weather/Natural Disasters
- Theft
- Road Closures
- Traffic
- Tolls

## Tasks
- Make the Graph
- Learn and Implement the travelling salesman problem on the Quantum Computer and the Classical System
- APIs for information
- Theft information exists?
- Predictive Alerts

## Thought process:
- Get a subgraph within a certain radius of all nodes that need to be delivered to
- Perform Max cut on the cities
- Repeat max cut until all trucks are accounted for
- Then provide the travelling salesman problem in each cluster, and compute the cost

*As of February 6th 2024, the API key present in this repository has been deactivated. The code will still run fine but only after a different API key is inserted*
