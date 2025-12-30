% Bag of Tasks Scheduling Model in MiniZinc
% Converted from Java Choco Solver implementation

include "globals.mzn";

% ============= PARAMETERS =============

% Number of nodes and data items
int: nb_nodes;
int: nb_data;
int:makespan;

int: factor = 1;

% Data 
array[1..nb_data] of int: data_sizes;
%array[1..nb_data] of int: starting_times;
array[1..nb_data] of int: nb_works;
array[1..nb_data] of int: work_duration;

% Node characteristics
array[1..nb_nodes] of int: bandwidths;     % MB/s
array[1..nb_nodes] of float: cpus;         % CPU units/s
array[1..nb_nodes] of float: node_free_timespan;


int: N = nb_nodes;
int: D = nb_data;

array[1..N, 1..D] of float: transfers_time;

% ============= HELPER FUNCTIONS =============

% Calculate transfer duration for data i on node j
function int: transfer_duration(int: i, int: j) =
  ceil(transfers_time[j,i]);
  %ceil(int2float(data_sizes[i]) / int2float(bandwidths[j]));

% Calculate work duration for data i on node j
function int: work_duration(int: i, int: j) =
  floor(int2float(work_duration[i]) * cpus[j]);

% ============= DECISION VARIABLES =============

% Transfer variables: for each node and data
array[1..N, 1..D] of var 0..makespan: transfer_start;
array[1..N, 1..D] of var 0..makespan: transfer_end;
array[1..N, 1..D] of var 0..1: transfer_height;


% Work variables: for each node, data, and work index
array[1..N, 1..D, 1..max(nb_works)] of var 0..makespan: work_start;
array[1..N, 1..D, 1..max(nb_works)] of var 0..makespan: work_end;
array[1..N, 1..D, 1..max(nb_works)] of var 0..1: work_height;

% Objective variable
var 0..makespan: objective_value;

% ============= CONSTRAINTS =============

% TRANSFER
% 1. Transfer task consistency: end = start + duration
constraint forall(j in 1..N, i in 1..D) (
  transfer_end[j,i] = transfer_start[j,i] + transfer_duration(i, j)
);

% 2. Transfer start time must respect data starting time
constraint forall(j in 1..N, i in 1..D) (
  transfer_start[j,i] >= node_free_timespan[j] %starting_times[i]
);

% WORK
% 3. Work task consistency: end = start + duration (only if work exists)
constraint forall(j in 1..N, i in 1..D, k in 1..nb_works[i]) (
  work_end[j,i,k] = work_start[j,i,k] + work_duration(i, j)
);

% 4. Work start time must respect data starting time
constraint forall(j in 1..N, i in 1..D, k in 1..nb_works[i]) (
  work_start[j,i,k] >= node_free_timespan[j] %starting_times[i]
);

% 5. Cumulative constraint for transfers on each node (capacity = 1)
constraint forall(j in 1..N) (
  cumulative(
    [transfer_start[j,i] | i in 1..D],
    [transfer_duration(i, j) | i in 1..D],
    [transfer_height[j,i] | i in 1..D],
    1
  )
);

% 6. At least one transfer per data item
constraint forall(i in 1..D) (
  sum(j in 1..N)(transfer_height[j,i]) >= 1
);

% 7. Work can start only after corresponding transfer finishes
constraint forall(j in 1..N, i in 1..D, k in 1..nb_works[i]) (
  work_start[j,i,k] >= transfer_end[j,i]
);

% 8. Regular constraint: pattern (0*)|(10*1+0*)
%    If transfer happens on a node, at least one work must happen
%    Works can only happen if transfer happened
constraint forall(j in 1..N, i in 1..D) (
  forall(k in 1..nb_works[i]) (
    work_height[j,i,k] <= transfer_height[j,i]
  )
);

% On the same node, works are in lex order
constraint forall(i in 1..D) (
  forall(j in 1..N, k in 1..nb_works[i]-1) (
    work_start[j,i,k] <= work_start[j,i,k+1]
  )
);

% TODO break symetries to make sure that the smallest work is executed on the smallest node
% constraint forall(d in 1..D, wi in 1..max(nb_works), wj in wi+1..max(nb_works)) (
%   lex_greatereq(
%     array1d(1..N, [work_height[j, d, wi] | j in 1..N]),
%     array1d(1..N, [work_height[j, d, wj] | j in 1..N])
%   )
% );

constraint forall(j in 1..N, i in 1..D) (
  transfer_height[j,i] <= sum(k in 1..nb_works[i])(work_height[j,i,k])
);

% 9. Cumulative constraint for works on each node (capacity = 1)
constraint forall(j in 1..N) (
  let {
    array[int] of var int: all_starts = 
      [work_start[j,i,k] | i in 1..D, k in 1..nb_works[i]],
    array[int] of int: all_durations = 
      [work_duration(i,j) | i in 1..D, k in 1..nb_works[i]],
    array[int] of var int: all_heights = 
      [work_height[j,i,k] | i in 1..D, k in 1..nb_works[i]]
  } in
  cumulative(all_starts, all_durations, all_heights, 1)
);

% 10. Each work must be executed exactly once (across all nodes)
constraint forall(i in 1..D, k in 1..nb_works[i]) (
  sum(j in 1..N)(work_height[j,i,k]) = 1
);

array [1..D] of var bool : is_transfered_more_than_once;
constraint forall(i in 1..D)(
  is_transfered_more_than_once[i] = (sum(j in 1..N)(transfer_height[j,i]) > 1)
);


% 11. Sum of works should be greater or equal to transfer duration (Utility)
constraint forall(j in 1..N, i in 1..D) (
  is_transfered_more_than_once[i] ->  transfer_height[j,i] * transfer_duration(i, j) <= factor * sum(k in 1..nb_works[i])(work_height[j,i,k] * work_duration(i,j))
%  (is_transfered_more_than_once[i] - 1) * factor * transfer_height[j,i] * transfer_duration(i, j) <= (is_transfered_more_than_once[i] - 1) * sum(k in 1..nb_works[i])(work_height[j,i,k] * work_duration(i,j))
);


% ============= OBJECTIVE =============

% Maximum Flow Time objective
% Flow time for data i = (completion time of data i) - (starting time of data i)

array[1..D] of var 0..makespan: completion_time;
array[1..D] of var 0..makespan: flow_time;


% Completion time is the maximum end time of all works for that data
constraint forall(i in 1..D) (
  completion_time[i] = max(j in 1..N, k in 1..nb_works[i])(
    work_height[j,i,k] * work_end[j,i,k]
  )
);

% Flow time calculation
constraint forall(i in 1..D) (
  flow_time[i] = completion_time[i] % - starting_times[i]
);

int: lbds = sum(data_sizes);
var int : data_transfer;
constraint data_transfer = sum(i in 1..D, j in 1..N)(transfer_height[j,i] * data_sizes[i]);



% % Objective: minimize maximum flow time
%constraint objective_value = max(flow_time);
constraint objective_value = sum(flow_time);
% constraint objective_value <= 852*1.05;


solve
::seq_search([
    int_search(work_height,input_order,indomain_min), 
    int_search(transfer_height, input_order,indomain_min),
    int_search(transfer_start, input_order, indomain_split),
    int_search(work_start, input_order, indomain_split)])
    
minimize objective_value;

%minimize 
%10000 * objective_value + data_transfer;
%data_transfer div objective_value;
%objective_value;
%data_transfer;
%satisfy;



output [ show(objective_value) ++ ";" ++ show(data_transfer) ++";\n" ]
