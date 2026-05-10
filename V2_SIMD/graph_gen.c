/**
 * Copyright (C) 2023 Polverino Alessandro
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 * 
 * Student: Polverino Alessandro 0622702352, a.polverino15@studenti.unisa.it
 * Lecturer: Prof. Francesco Moscato, fmoscato@unisa.it
 * Assignment: HPC-Parallel-Bellman-Ford-Implementation
 * 
 * Purpose of this file:
 * This file uses graph.c to generate random graphs and save them to files.
 * 
*/

#include "./include/graph.h"

#include <stdio.h>


int main() {
  int V1 = 10;
  int V2 = 100;
  int V3 = 1000;
  int V4 = 5000;

  // density values has impact on the number of operations
  // if no edge exists between two vertices, the weight is set to WEIGHT_INFINITY
  // therefore the bf algorithm will not consider that edge 
  // (only an if statement is executed instead of a relax operation)
  float D1 = 0.25;
  float D2 = 0.5;
  float D3 = 0.75;
  float D4 = 1;


  graph_t* GV1D1 = get_empty_graph();
  generate_random_graph(V1, D1, GV1D1);
  save_graph_to_file(GV1D1, "./data/inputs/graph_V1_D1.bin");
  destroy_graph(GV1D1);

  graph_t* GV1D2 = get_empty_graph();
  generate_random_graph(V1, D2, GV1D2);
  save_graph_to_file(GV1D2, "./data/inputs/graph_V1_D2.bin");
  destroy_graph(GV1D2);

  graph_t* GV1D3 = get_empty_graph();
  generate_random_graph(V1, D3, GV1D3);
  save_graph_to_file(GV1D3, "./data/inputs/graph_V1_D3.bin");
  destroy_graph(GV1D3);

  graph_t* GV1D4 = get_empty_graph();
  generate_random_graph(V1, D4, GV1D4);
  save_graph_to_file(GV1D4, "./data/inputs/graph_V1_D4.bin");
  destroy_graph(GV1D4);

  graph_t* GV2D1 = get_empty_graph();
  generate_random_graph(V2, D1, GV2D1);
  save_graph_to_file(GV2D1, "./data/inputs/graph_V2_D1.bin");
  destroy_graph(GV2D1);
  
  graph_t* GV2D2 = get_empty_graph();
  generate_random_graph(V2, D2, GV2D2);
  save_graph_to_file(GV2D2, "./data/inputs/graph_V2_D2.bin");
  destroy_graph(GV2D2);

  graph_t* GV2D3 = get_empty_graph();
  generate_random_graph(V2, D3, GV2D3);
  save_graph_to_file(GV2D3, "./data/inputs/graph_V2_D3.bin");
  destroy_graph(GV2D3);

  graph_t* GV2D4 = get_empty_graph();
  generate_random_graph(V2, D4, GV2D4);
  save_graph_to_file(GV2D4, "./data/inputs/graph_V2_D4.bin");
  destroy_graph(GV2D4);

  graph_t* GV3D1 = get_empty_graph();
  generate_random_graph(V3, D1, GV3D1);
  save_graph_to_file(GV3D1, "./data/inputs/graph_V3_D1.bin");
  destroy_graph(GV3D1);

  graph_t* GV3D2 = get_empty_graph();
  generate_random_graph(V3, D2, GV3D2);
  save_graph_to_file(GV3D2, "./data/inputs/graph_V3_D2.bin");
  destroy_graph(GV3D2);

  graph_t* GV3D3 = get_empty_graph();
  generate_random_graph(V3, D3, GV3D3);
  save_graph_to_file(GV3D3, "./data/inputs/graph_V3_D3.bin");
  destroy_graph(GV3D3);

  graph_t* GV3D4 = get_empty_graph();
  generate_random_graph(V3, D4, GV3D4);
  save_graph_to_file(GV3D4, "./data/inputs/graph_V3_D4.bin");
  destroy_graph(GV3D4);

  graph_t* GV4D1 = get_empty_graph();
  generate_random_graph(V4, D1, GV4D1);
  save_graph_to_file(GV4D1, "./data/inputs/graph_V4_D1.bin");
  destroy_graph(GV4D1);

  graph_t* GV4D2 = get_empty_graph();
  generate_random_graph(V4, D2, GV4D2);
  save_graph_to_file(GV4D2, "./data/inputs/graph_V4_D2.bin");
  destroy_graph(GV4D2);

  graph_t* GV4D3 = get_empty_graph();
  generate_random_graph(V4, D3, GV4D3);
  save_graph_to_file(GV4D3, "./data/inputs/graph_V4_D3.bin");
  destroy_graph(GV4D3);

  graph_t* GV4D4 = get_empty_graph();
  generate_random_graph(V4, D4, GV4D4);
  save_graph_to_file(GV4D4, "./data/inputs/graph_V4_D4.bin");
  destroy_graph(GV4D4);
}
