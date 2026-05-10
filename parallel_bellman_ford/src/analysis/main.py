
#   Copyright (C) 2023 Polverino Alessandro
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#   Student: Polverino Alessandro 0622702352, a.polverino15@studenti.unisa.it
#   Lecturer: Prof. Francesco Moscato, fmoscato@unisa.it
#   Assignment: HPC-Parallel-Bellman-Ford-Implementation
#
#   Purpose of this file:
#   This file contains logic to analyse the results of the execution of the program.

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import pandas as pd
import analysis
import matplotlib.pyplot as plt
import dataframe_image as dfi
import numpy as np


DATA_PATH = path = os.path.join(os.path.dirname(__file__), '../../data/')
OUTPUT_PATH = path = os.path.join(DATA_PATH, 'outputs/')

# get the csv files with execution data
# threads_num,cuda_threads_num,processes_num,program,vertices_num,density,cuda_grid_size,cuda_block_size,percent_on_device,program_seconds,bf_seconds,output_filename,has_negative_cycles
files = analysis.get_csv_files(DATA_PATH)

complete_df = None

for file in files:
    output_df = pd.read_csv(file, sep=',')
    
    # get distinct values of vertices_num and density
    vertices_num = output_df['vertices_num'].unique()
    density = output_df['density'].unique()
    # get all the combinations of vertices_num and density
    vert_dens_combinations = [(v, d) for v in vertices_num for d in density]
    # get different dfs for each combination
    vert_dens_dfs = [output_df[(output_df['vertices_num'] == v) & (output_df['density'] == d)] for v, d in vert_dens_combinations]
    for input_type_result_df in vert_dens_dfs:
        # for each df 
        # get the serial row
        serial_row = input_type_result_df[input_type_result_df['program'].str.contains('serial')].iloc[0]
        serial_output_filename = serial_row.output_filename.split('/')[-1]
        serial_output_path = os.path.join(OUTPUT_PATH, serial_output_filename)
        # get the parallel rows
        parallel_df = input_type_result_df[input_type_result_df['program'] != serial_row.program]
        # check correctness of each parallel row
        incorrect = 0
        for idx, par_row in parallel_df.iterrows(): 
            # check if output_filename exists and is not ""
            if not par_row.output_filename or not os.path.exists(os.path.join(OUTPUT_PATH, par_row.output_filename.split('/')[-1])):
                continue
            par_row_output_filename = par_row.output_filename.split('/')[-1]
            par_row_output_path = os.path.join(OUTPUT_PATH, par_row_output_filename)
            correct = analysis.check_file_correct(serial_output_path, par_row_output_path)
            if not correct:
                incorrect += 1
                print(f"ERROR: {par_row.program} {par_row.output_filename} {serial_output_filename} | on G with {par_row.vertices_num} vertices and density {par_row.density}")
        if incorrect > 0:
            print(f"ERROR: some results ({incorrect}) are not correct (more details above)")
            exit(1)
            
        # must exlude from parallel_df, between rows with same program name, the ones with times too large or too small
        # because they are not reliable results. Calc std and mean of the times and exclude the ones that are too far from the mean
        clean_parallel_df = pd.DataFrame(columns=parallel_df.columns)
        program_names = parallel_df['program'].unique()
        for program_name in program_names:
            # for each program name in the parallels i need to get executions with same
            # parameters, exclude the ones with times too large or too small times
            # and calc the mean of the times
            program_df = parallel_df[parallel_df['program'] == program_name].copy()
            if len(program_df) > 1:
                program_df = program_df[(program_df['program_seconds'] < program_df['program_seconds'].mean() + 3 * program_df['program_seconds'].std()) & (program_df['program_seconds'] > program_df['program_seconds'].mean() - 3 * program_df['program_seconds'].std())]
            program_df = program_df.groupby(['threads_num', 'cuda_threads_num', 'processes_num', 'program', 'cuda_grid_size','cuda_block_size','percent_on_device']).mean(numeric_only=True).reset_index()
            clean_parallel_df = pd.concat([clean_parallel_df, program_df], ignore_index=True)
                    
                    
        # for each parallel row, calc speedup and efficiency
        clean_parallel_df['program_speedup'] = serial_row['program_seconds'] / clean_parallel_df['program_seconds']
        clean_parallel_df['bf_speedup'] = serial_row['bf_seconds'] / clean_parallel_df['bf_seconds']
        
        def calc_prog_efficiency(row):
            if row['threads_num'] > 1 and row['processes_num'] == 1: # OMP
                return row['program_speedup'] / row['threads_num']
            elif row['threads_num'] == 0 and row['processes_num'] > 1: # MPI
                return row['program_speedup'] / row['processes_num']
            elif row['threads_num'] > 1 and row['processes_num'] > 1: # GOAL_1
                return row['program_speedup'] / (row['threads_num'] * row['processes_num'])
            else:
                return np.nan
            
        def calc_bf_efficiency(row):
            if row['threads_num'] > 1 and row['processes_num'] == 1: # OMP
                return row['bf_speedup'] / row['threads_num']
            elif row['threads_num'] == 0 and row['processes_num'] > 1: # MPI
                return row['bf_speedup'] / row['processes_num']
            elif row['threads_num'] > 1 and row['processes_num'] > 1: # GOAL_1
                return row['bf_speedup'] / (row['threads_num'] * row['processes_num'])
        
        clean_parallel_df['program_efficiency'] = clean_parallel_df.apply(calc_prog_efficiency, axis=1)
        clean_parallel_df['bf_efficiency'] = clean_parallel_df.apply(calc_bf_efficiency, axis=1)

        # dropping useless columns
        clean_parallel_df.drop(columns=['has_negative_cycles', 'output_filename'], inplace=True)
                
        opt = file.split("/")[-1].split(".")[0].split("_")[-1]

        out_analytics_filename = f"analytics_{opt}_NumV_{serial_row.vertices_num}_NumD_{serial_row.density}.csv"
        clean_parallel_df.to_csv(os.path.join(DATA_PATH, 'analytics', out_analytics_filename), index=False)
        
        
        title = f"({opt}) VERTICES: {serial_row.vertices_num} DENSITY: {serial_row.density}"
        
        # PLOTS
        omp_df = clean_parallel_df[clean_parallel_df['program'].str.contains('OMP')]
        mpi_df = clean_parallel_df[clean_parallel_df['program'].str.contains('MPI')]
        cuda_df = clean_parallel_df[clean_parallel_df['program'].str.contains('CUDA')]
        cuda_df.loc[:, ('cuda_threads_num')] = cuda_df['cuda_threads_num'] 
        goal_1_df = clean_parallel_df[clean_parallel_df['program'].str.contains('GOAL_1')]
        goal_2_df = clean_parallel_df[clean_parallel_df['program'].str.contains('GOAL_2')]
        
        if len(omp_df) > 0 or len(mpi_df) > 0 or len(cuda_df) > 0:
            fig, ax = plt.subplots()
            if len(omp_df) > 0:
                ax.plot(omp_df['threads_num'], omp_df['bf_speedup'], label='OMP bf speedup')
            if len(mpi_df) > 0:
                ax.plot(mpi_df['processes_num'], mpi_df['bf_speedup'], label='MPI bf speedup')
            if len(cuda_df) > 0:
                # plot cuda on different horizontal axis
                ax2 = ax.twiny()
                ax2.plot(cuda_df['cuda_threads_num'], cuda_df['bf_speedup'], label='CUDA bf speedup', marker='o', markersize=5, color="c")
                ax2.set_xlabel("CUDA Threads")
                ax2.legend()

            # get max speedup between the three dfs
            max_speedup = max([omp_df['bf_speedup'].max(), mpi_df['bf_speedup'].max(), cuda_df['bf_speedup'].max()])
            ax.axhline(y=max_speedup, color='g', linestyle='--', label='max speedup')    
            if max_speedup >= 0.5:
                # useless to plot 1x speedup if max speedup is less than 0.5,
                # would compress the plot too much to be readable
                ax.axhline(y=1, color='r', linestyle='--', label='1x speedup (serial)')
            ax.legend()
            # set title
            plt.title(title)
            ax.set_xlabel("Threads/Processes")
            ax.set_ylabel("Speedup")
            plt.savefig(os.path.join(DATA_PATH, 'plots', f"plot_speedup_only_versions_{opt}_NumV_{serial_row.vertices_num}_NumD_{serial_row.density}.png"), bbox_inches="tight")
            plt.close()
        
        if len(goal_1_df) > 0:
            plt.figure()
            fig, ax = plt.subplots()
            plt.title(title)
            ax.plot(goal_1_df['threads_num'], goal_1_df['bf_speedup'], label='GOAL_1 program speedup')
            for idx, row in goal_1_df.iterrows():
                ax.annotate(row.processes_num, (row.threads_num, row.bf_speedup))
            ax.hlines(y=goal_1_df['bf_speedup'].max(), xmin=goal_1_df['threads_num'].min(), xmax=goal_1_df['threads_num'].max(), color='g', linestyle='--', label='max speedup')
            if goal_1_df['bf_speedup'].max() >= 0.5:
                # useless to plot 1x speedup if max speedup is less than 0.5,
                # would compress the plot too much to be readable
                ax.hlines(y=1, xmin=goal_1_df['threads_num'].min(), xmax=goal_1_df['threads_num'].max(), color='r', linestyle='--', label='1x speedup (serial)')
            plt.xlabel("Threads")
            plt.ylabel("Speedup")
            plt.legend()
            plt.savefig(os.path.join(DATA_PATH, 'plots', f"plot_speedup_goal_1_{opt}_NumV_{serial_row.vertices_num}_NumD_{serial_row.density}.png"))
            plt.close()
        
        if len(goal_2_df) > 0:
            plt.figure()
            fig, ax = plt.subplots()
            plt.title(title)
            ax.plot(goal_2_df['threads_num'], goal_2_df['bf_speedup'], label='GOAL_2 program speedup')
            for idx, row in goal_2_df.iterrows():
                ax.annotate(row.percent_on_device, (row.threads_num, row.bf_speedup))
            ax.hlines(y=goal_2_df['bf_speedup'].max(), xmin=goal_2_df['threads_num'].min(), xmax=goal_2_df['threads_num'].max(), color='g', linestyle='--', label='max speedup')
            if goal_2_df['bf_speedup'].max() >= 0.5:
                ax.hlines(y=1, xmin=goal_2_df['threads_num'].min(), xmax=goal_2_df['threads_num'].max(), color='r', linestyle='--', label='1x speedup (serial)')
            plt.xlabel("Threads")
            plt.ylabel("Speedup")
            plt.legend()
            plt.savefig(os.path.join(DATA_PATH, 'plots', f"plot_speedup_goal_2_{opt}_NumV_{serial_row.vertices_num}_NumD_{serial_row.density}.png"))
            plt.close()

        serial_row = serial_row.drop(['has_negative_cycles', 'output_filename'])
        top_executions = clean_parallel_df.sort_values(by=['bf_speedup'], ascending=False).head(5).copy()
        top_executions = top_executions._append(serial_row.to_dict(), ignore_index=True).sort_values(by=['bf_speedup'], ascending=False)
        
        worst_executions = clean_parallel_df.sort_values(by=['bf_speedup'], ascending=True).head(5).copy()
        worst_executions = worst_executions._append(serial_row.to_dict(), ignore_index=True).sort_values(by=['bf_speedup'], ascending=True)
        
        dfi.export(top_executions, os.path.join(DATA_PATH, 'tables', f"top_executions_{opt}_NumV_{serial_row.vertices_num}_NumD_{serial_row.density}.png"), table_conversion='matplotlib')
        dfi.export(worst_executions, os.path.join(DATA_PATH, 'tables', f"worst_executions_{opt}_NumV_{serial_row.vertices_num}_NumD_{serial_row.density}.png"), table_conversion='matplotlib')
        
        clean_parallel_df['program'] = clean_parallel_df['program'].apply(lambda x: f"({opt}) {x}")
        if complete_df is None:
            complete_df = clean_parallel_df.copy()
        else:
            complete_df = pd.concat([complete_df, clean_parallel_df], ignore_index=True)

overall_top_executions = complete_df.sort_values(by=['bf_speedup'], ascending=False).head(5).copy()
overall_worst_executions = complete_df.sort_values(by=['bf_speedup'], ascending=True).head(5).copy()

dfi.export(overall_top_executions, os.path.join(DATA_PATH, 'tables', f"overall_top_executions.png"), table_conversion='matplotlib')
dfi.export(overall_worst_executions, os.path.join(DATA_PATH, 'tables', f"overall_worst_executions.png"), table_conversion='matplotlib')

# saving csv with all the executions ordered by (vertices_num, density), speedup
complete_df = complete_df.sort_values(by=['vertices_num', 'density', 'bf_speedup'], ascending=False)
complete_df.to_csv(os.path.join(DATA_PATH, 'analytics', 'complete_analytics.csv'), index=False)

# now considering only the executions with vertices_num >= 1000 (V3)
complete_df = complete_df[complete_df['vertices_num'] >= 1000]
minV3_top_executions = complete_df.sort_values(by=['bf_speedup'], ascending=False).head(5).copy()
minV3_worst_executions = complete_df.sort_values(by=['bf_speedup'], ascending=True).head(5).copy()

dfi.export(minV3_top_executions, os.path.join(DATA_PATH, 'tables', f"minV3_top_executions.png"), table_conversion='matplotlib')
dfi.export(minV3_worst_executions, os.path.join(DATA_PATH, 'tables', f"minV3_worst_executions.png"), table_conversion='matplotlib')
