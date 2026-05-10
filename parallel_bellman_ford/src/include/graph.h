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
 * This file contains the headers for utility/graph.c.
 * 
*/


#ifndef GRAPH_H
#define GRAPH_H

#include <stdbool.h>

typedef struct {
  unsigned int V; // number of vertices
  float D; // density
  int* edges;
  bool has_negative_cycle;
} graph_t;

#define DISTANCE_INFINITY 1000000
#define DISTANCE_MINUS_INFINITY -1000000

graph_t* get_empty_graph();
void destroy_graph(graph_t* G);
void generate_random_graph(int V, float D, graph_t* G);
void save_graph_to_file(graph_t *G, char *filename);
void read_graph_from_file(graph_t *G, char *filename);
void print_graph(graph_t *G);

#endif // GRAPH_H