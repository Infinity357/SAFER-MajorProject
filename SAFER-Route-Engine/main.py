import os
import math
import random
import time

import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt



PLACE = "Solan, Himachal Pradesh, India"

RANDOM_SEED = 42

# Simulated road-condition probabilities
ROAD_CONDITIONS = {
    "normal": 0.65,
    "damaged": 0.15,
    "flooded": 0.12,
    "blocked": 0.08
}

# Vehicle-specific routing profiles
VEHICLE_PROFILES = {
    "fire_truck": {
        "risk_sensitivity": 2.5,
        "damage_sensitivity": 5.0
    },

    "ambulance": {
        "risk_sensitivity": 4.0,
        "damage_sensitivity": 5.0
    },

    "volunteer": {
        "risk_sensitivity": 5.0,
        "damage_sensitivity": 6.0
    },

    "rescue": {
        "risk_sensitivity": 3.0,
        "damage_sensitivity": 4.0
    }
}

# Emergency resources
RESOURCES = [
    {
        "name": "Fire Truck 01",
        "type": "fire_truck",
        "color": "red"
    },
    {
        "name": "Ambulance 01",
        "type": "ambulance",
        "color": "green"
    },
    {
        "name": "Volunteer Team 01",
        "type": "volunteer",
        "color": "blue"
    },
    {
        "name": "Rescue Team 01",
        "type": "rescue",
        "color": "purple"
    }
]


#utils
def print_header(title):

    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def haversine(lat1, lon1, lat2, lon2):

    earth_radius = 6371000

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    dlat = lat2 - lat1
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return (
        2 * earth_radius
        * math.asin(math.sqrt(a))
    )

#dowbload road netowrk
def download_road_network(place):

    print_header("SAFER - MULTI-AGENCY ROUTE PLANNING")

    print("\n[1] Downloading road network...")
    print(f"    Location: {place}")

    G = ox.graph_from_place(
        place,
        network_type="drive"
    )

    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    print(f"    Nodes: {len(G.nodes):,}")
    print(f"    Edges: {len(G.edges):,}")

    return G

#generatiion of disaster condition
def generate_disaster_data(G):

    print("\n[2] Generating simulated disaster conditions...")

    rng = random.Random(RANDOM_SEED)

    # Same physical road gets the same condition
    # in both directions.

    segment_conditions = {}

    counts = {
        "normal": 0,
        "damaged": 0,
        "flooded": 0,
        "blocked": 0
    }

    for u, v, key, data in G.edges(
        keys=True,
        data=True
    ):

        segment_id = tuple(sorted((u, v)))

        if segment_id not in segment_conditions:

            condition = rng.choices(
                population=list(ROAD_CONDITIONS.keys()),
                weights=list(ROAD_CONDITIONS.values()),
                k=1
            )[0]

            if condition == "normal":

                risk = rng.uniform(0.0, 0.15)
                damage = rng.uniform(0.0, 0.05)

            elif condition == "damaged":

                risk = rng.uniform(0.2, 0.55)
                damage = rng.uniform(0.25, 0.65)

            elif condition == "flooded":

                risk = rng.uniform(0.65, 0.95)
                damage = rng.uniform(0.1, 0.4)

            else:

                risk = 1.0
                damage = 1.0

            segment_conditions[segment_id] = (
                condition,
                risk,
                damage
            )

        condition, risk, damage = segment_conditions[
            segment_id
        ]

        data["condition"] = condition
        data["risk"] = risk
        data["damage"] = damage
        data["blocked"] = condition == "blocked"

        counts[condition] += 1

    print("\n    Road condition distribution:")

    for condition, count in counts.items():

        print(
            f"    {condition.capitalize():12}: {count}"
        )

    return G


#calculating edge for each vehical
def calculate_edge_weights(G):

    print("\n[3] Calculating vehicle-specific edge weights...")

    for u, v, key, data in G.edges(
        keys=True,
        data=True
    ):

        travel_time = data.get("travel_time", 60)

        risk = data["risk"]
        damage = data["damage"]

        for vehicle, profile in VEHICLE_PROFILES.items():

            alpha = profile["risk_sensitivity"]
            beta = profile["damage_sensitivity"]

            data[f"cost_{vehicle}"] = (
                travel_time
                * (
                    1
                    + alpha * risk
                    + beta * damage
                )
            )

    print("    All vehicle-specific weights calculated.")

    return G


#creating routable graph
def create_routable_graph(G):

    print("\n[4] Removing completely blocked roads...")

    H = G.copy()

    blocked_edges = [
        (u, v, key)
        for u, v, key, data in H.edges(
            keys=True,
            data=True
        )
        if data["blocked"]
    ]

    H.remove_edges_from(blocked_edges)

    print(f"    Removed blocked edges: {len(blocked_edges)}")
    print(f"    Available edges: {len(H.edges):,}")

    return H


#selecting random resource and organisation loctions
def select_demo_locations(G):

    print("\n[5] Selecting emergency locations...")

    nodes = list(G.nodes)

    if len(nodes) < len(RESOURCES) + 1:
        raise ValueError("Not enough road network nodes.")

    # Select the geographical centre as the disaster location.

    avg_lat = sum(
        G.nodes[n]["y"] for n in nodes
    ) / len(nodes)

    avg_lon = sum(
        G.nodes[n]["x"] for n in nodes
    ) / len(nodes)

    destination = min(
        nodes,
        key=lambda n: haversine(
            avg_lat,
            avg_lon,
            G.nodes[n]["y"],
            G.nodes[n]["x"]
        )
    )

    selected = [destination]

    # Select geographically distributed starting nodes.

    for resource in RESOURCES:

        available = [
            n for n in nodes
            if n not in selected
        ]

        start_node = max(
            available,
            key=lambda n: min(
                haversine(
                    G.nodes[n]["y"],
                    G.nodes[n]["x"],
                    G.nodes[s]["y"],
                    G.nodes[s]["x"]
                )
                for s in selected
            )
        )

        resource["node"] = start_node

        resource["location"] = (
            G.nodes[start_node]["y"],
            G.nodes[start_node]["x"]
        )

        selected.append(start_node)

    print(f"\n    Disaster destination: {destination}")

    for resource in RESOURCES:

        print(
            f"    {resource['name']:22} "
            f"Start node: {resource['node']}"
        )

    return destination


# parllel edge for vehicle
def get_best_edge_for_vehicle(
    G,
    u,
    v,
    vehicle_type
):

    edges = G.get_edge_data(u, v)

    if not edges:
        raise ValueError(f"Missing edge: {u} -> {v}")

    weight = f"cost_{vehicle_type}"

    return min(
        edges.values(),
        key=lambda data: data[weight]
    )


#resource route calculation
def calculate_resource_route(
    G,
    resource,
    destination
):

    vehicle_type = resource["type"]

    start_node = resource["node"]

    weight = f"cost_{vehicle_type}"

    start_time = time.perf_counter()

    route = nx.shortest_path(
        G,
        source=start_node,
        target=destination,
        weight=weight,
        method="dijkstra"
    )

    execution_time = (
        time.perf_counter() - start_time
    ) * 1000

    return route, execution_time


#route statistics calculation
def calculate_route_statistics(
    G,
    route,
    vehicle_type
):

    total_distance = 0
    total_time = 0
    total_cost = 0
    risk_distance_sum = 0

    conditions = {
        "normal": 0,
        "damaged": 0,
        "flooded": 0
    }

    for u, v in zip(route[:-1], route[1:]):

        data = get_best_edge_for_vehicle(
            G,
            u,
            v,
            vehicle_type
        )

        length = data.get("length", 0)

        total_distance += length

        total_time += data.get(
            "travel_time", 0
        )

        total_cost += data[
            f"cost_{vehicle_type}"
        ]

        risk_distance_sum += (
            data["risk"] * length
        )

        condition = data["condition"]

        if condition in conditions:
            conditions[condition] += 1

    average_risk = (
        risk_distance_sum / total_distance
        if total_distance > 0
        else 0
    )

    return {
        "distance_km": total_distance / 1000,
        "travel_time_min": total_time / 60,
        "risk_adjusted_cost": total_cost,
        "average_risk": average_risk,
        "conditions": conditions,
        "edge_count": len(route) - 1
    }


#displayinf indvidual route
def display_resource_route(
    G,
    resource,
    route,
    destination,
    execution_time
):

    stats = calculate_route_statistics(
        G,
        route,
        resource["type"]
    )

    print_header(resource["name"].upper())

    print(f"Resource type:       {resource['type']}")
    print(f"Starting node:       {resource['node']}")
    print(f"Destination node:    {destination}")

    print("\nRoute statistics:")

    print(
        f"Distance:            "
        f"{stats['distance_km']:.2f} km"
    )

    print(
        f"Base travel ETA:     "
        f"{stats['travel_time_min']:.2f} minutes"
    )

    print(
        f"Risk-adjusted cost:  "
        f"{stats['risk_adjusted_cost']:.2f}"
    )

    print(
        f"Average road risk:   "
        f"{stats['average_risk']:.3f}"
    )

    print(
        f"Road segments:       "
        f"{stats['edge_count']}"
    )

    print(
        f"Execution time:      "
        f"{execution_time:.3f} ms"
    )

    print("\nRoad conditions along route:")

    for condition, count in stats["conditions"].items():

        print(f"    {condition.capitalize():12}: {count}")

    print("\nRoute nodes:")

    for i, node in enumerate(route):

        if i >= 12:
            print("    ...")
            break

        print(f"    {i + 1}. Node {node}")

    return stats


#routing every organisations
def run_multi_resource_routing(
    G,
    destination
):

    print_header("MULTI-AGENCY EMERGENCY DEPLOYMENT")

    results = []

    for resource in RESOURCES:

        print(
            f"\nCalculating route for {resource['name']}..."
        )

        try:

            route, execution_time = calculate_resource_route(
                G,
                resource,
                destination
            )

            stats = display_resource_route(
                G,
                resource,
                route,
                destination,
                execution_time
            )

            results.append({
                "name": resource["name"],
                "type": resource["type"],
                "color": resource["color"],
                "route": route,
                "execution_time": execution_time,
                **stats
            })

        except nx.NetworkXNoPath:

            print(
                f"    No route available for {resource['name']}."
            )

    return results


#final summary of response
def display_final_summary(results):

    print_header("FINAL EMERGENCY RESPONSE SUMMARY")

    print(
        f"{'RESOURCE':22}"
        f"{'DISTANCE':12}"
        f"{'ETA':12}"
        f"{'RISK':10}"
    )

    print("-" * 56)

    for result in results:

        print(
            f"{result['name']:22}"
            f"{result['distance_km']:8.2f} km "
            f"{result['travel_time_min']:7.2f} min "
            f"{result['average_risk']:8.3f}"
        )

    if results:

        earliest = min(
            results,
            key=lambda r: r["travel_time_min"]
        )

        print("\nEarliest estimated arrival:")
        print(f"    {earliest['name']}")
        print(
            f"    {earliest['travel_time_min']:.2f} minutes"
        )

    print("\nSuccessfully routed:", len(results))
    print("Total resources:", len(RESOURCES))


#save route map for multiple oragnisations
def save_route_map(G, results, destination):

    print("\n[7] Generating multi-agency route map...")

    fig, ax = ox.plot_graph(
        G,
        node_size=0,
        edge_color="#B8B8B8",
        edge_linewidth=0.5,
        bgcolor="white",
        show=False,
        close=False
    )

    # Plot all resource routes on the same map.

    for result in results:

        ox.plot_graph_route(
            G,
            result["route"],
            ax=ax,
            route_color=result["color"],
            route_linewidth=2.5,
            node_size=0,
            show=False,
            close=False
        )

    # Mark disaster destination.

    destination_x = G.nodes[destination]["x"]
    destination_y = G.nodes[destination]["y"]

    ax.scatter(
        destination_x,
        destination_y,
        c="black",
        marker="*",
        s=220,
        zorder=10,
        label="Disaster Location"
    )

    # Mark starting points.

    for resource in RESOURCES:

        node = resource["node"]

        ax.scatter(
            G.nodes[node]["x"],
            G.nodes[node]["y"],
            c=resource["color"],
            marker="o",
            s=65,
            zorder=9,
            label=resource["name"]
        )

    ax.legend(loc="best", fontsize=8)

    os.makedirs("output", exist_ok=True)

    output_path = "output/multi_agency_routes.png"

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"    Map saved: {output_path}")




def main():

    # Download road network
    G = download_road_network(PLACE)

    # Generate dummy road conditions
    G = generate_disaster_data(G)

    # Calculate vehicle-specific weights
    G = calculate_edge_weights(G)

    # Remove blocked roads
    H = create_routable_graph(G)

    # Select resource locations from the available graph
    destination = select_demo_locations(H)

    # Calculate routes for all emergency agencies
    results = run_multi_resource_routing(
        H,
        destination
    )

    # Display final summary
    display_final_summary(results)

    # Save combined route visualization
    if results:

        save_route_map(
            H,
            results,
            destination
        )

    print_header("SAFER EXECUTION COMPLETED")


if __name__ == "__main__":
    main()
