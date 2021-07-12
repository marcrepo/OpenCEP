class TabuSearch:

    def __init__(self, initial_solution, max_tabu_list_len,stopping_criterion, neighborhood, score):
        """
        Stopping criterion as function that returns True iff it is ok to finish search.
        """
        self.curr_sol = initial_solution
        self.best_sol = initial_solution
        self.best_cost = score(initial_solution)
        self.tabu_list = []
        self.tabu_set = set()
        self.max_tabu_list_len = max_tabu_list_len
        self.stopping_criterion = stopping_criterion
        self.neighborhood = neighborhood
        self.score = score

    def run(self):
        while not self.stopping_criterion():
            # get all of the neighbors
            neighbors = self.neighborhood(self.curr_sol)

            # find all neighbors that are not part of the Tabu list
            unvisited_neighbors = [neighbor for neighbor in neighbors if neighbor not in self.tabu_set]
            #break if we have no more neighbors to explore
            if len(unvisited_neighbors) == 0:
                break
            # pick the best neighbor solution
            new_solution = sorted(unvisited_neighbors, key=lambda sol: self.score(sol))[0]
            # get the cost between the two solutions
            cost_diff = new_solution.cost-self.best_sol.cost
            # if the new solution is better,
            # update the best solution with the new solution
            if cost_diff <= 0:
                self.best_sol = new_solution
            # update the current solution with the new solution
            self.curr_sol = new_solution
            # adding new_solution to tabu list and if list is to long we will delete first element
            self.tabu_set.add(frozenset(new_solution.get_tabu_list_store_info()))
            self.tabu_list.append(frozenset(new_solution.get_tabu_list_store_info()))
            if len(self.tabu_list) > self.max_tabu_list_len:
                info_to_remove_from_tabu_set = self.tabu_list.pop(0)
                self.tabu_set.remove(info_to_remove_from_tabu_set)


            # return best solution found
        return self.best_sol.relevant_part_to_return()
