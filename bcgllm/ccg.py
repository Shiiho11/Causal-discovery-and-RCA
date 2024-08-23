import random

import networkx as nx

from bcgllm.llm import LLMClient


def generate_variables_description_text(variables_description: dict) -> str:
    text_lines = []
    for variable, description in variables_description.items():
        if description:
            text_lines.append(f'{variable}: {description}')
        else:
            text_lines.append(f'{variable}')
    if text_lines:
        return '\n'.join(text_lines)
    else:
        return ''


def generate_known_causal_relationship_text(causal_graph: nx.DiGraph) -> str:
    text_lines = []
    for node in causal_graph.nodes:
        successors = list(causal_graph.successors(node))
        if successors:
            successors_str_list = []
            for successor in successors:
                successors_str_list.append(str(successor))
            text_lines.append(f'{node} causes ' + ', '.join(successors_str_list))
    if text_lines:
        return '\n'.join(text_lines)
    else:
        return ''


def generate_request_text(domain: str, domain_expert: str, variables_description: dict, background, current_node,
                          current_graph: nx.DiGraph) -> str:
    text_blocks = []
    if domain_expert:
        text_blocks.append(f'You are a helpful assistant to a {domain_expert}.')
    else:
        text_blocks.append(f'You are a helpful assistant in the {domain} field.')
    if background:
        text_blocks.append('background:\n' + background)
    text_blocks.append(f'The following factors are key variables related to {domain} which have various causal effects on each other. '
                       f'Our goal is to construct a causal graph between these variables.')
    text_blocks.append(generate_variables_description_text(variables_description))

    if list(current_graph.edges):
        text_blocks.append('Given the known causal relationship as follows:')
        text_blocks.append(generate_known_causal_relationship_text(current_graph))

    text_blocks.append(f'You need to identify the causal relationship between variables. '
                       f'Select the variables that directly cause {current_node}.\n'
                       f'Think step by step. Then, provide your final answer (variable names only) within the tags '
                       f'<Answer>...</Answer>.')
    text = '\n\n'.join(text_blocks)
    return text


def get_predecessors_from_response(response: str, variables: list) -> list:
    start_index = response.index('<Answer>') + 8
    end_index = response.index('</Answer>')
    answer = response[start_index:end_index]
    predecessors = []
    for variable in variables:
        if str(variable).lower() in answer.lower():
            predecessors.append(variable)
    return predecessors


class CCG:

    def __init__(self, llm_client: LLMClient, domain: str, variables_description: dict, domain_expert=None,
                 background=None, known_causal_graph: nx.DiGraph = None,
                 start_node=None, full_visit=False):
        self.llm_client = llm_client
        self.domain = domain
        self.domain_expert = domain_expert
        self.variables_description = variables_description
        self.background = background
        if known_causal_graph:
            self.current_graph = known_causal_graph.copy()
        else:
            self.current_graph = nx.DiGraph()
        self.current_node = None
        self.visit_queue = []
        self.visited_nodes = []
        self.full_visit = full_visit

        if start_node is None:
            start_node = random.choice(list(variables_description.keys()))
        self.visit_queue.append(start_node)

        self.visit_counter = 0
        self.causal_graph = None

    def construct(self) -> nx.DiGraph:
        while self.visit_queue:
            self.current_node = self.visit_queue.pop(0)
            self.visited_nodes.append(self.current_node)
            predecessors = self.identify_predecessors_with_llm()
            for predecessor in predecessors:
                if not self.add_predecessor_has_cycle(predecessor):
                    self.current_graph.add_edge(predecessor, self.current_node)
                    if predecessor not in self.visited_nodes and predecessor not in self.visit_queue:
                        self.visit_queue.append(predecessor)

            if self.full_visit and not self.visit_queue:
                unvisited_nodes = [var for var in self.variables_description.keys() if var not in self.visited_nodes]
                if unvisited_nodes:
                    self.visit_queue.append(random.choice(unvisited_nodes))

        return self.current_graph

    def add_predecessor_has_cycle(self, predecessor):
        new_dag = self.current_graph.copy()
        new_dag.add_edge(predecessor, self.current_node)
        cycles = nx.recursive_simple_cycles(new_dag)
        if cycles:
            return True
        else:
            return False

    def identify_predecessors_with_llm(self, retry=5) -> list:
        self.visit_counter += 1
        request_text = generate_request_text(self.domain, self.domain_expert, self.variables_description,
                                             self.background, self.current_node, self.current_graph)
        predecessors = None
        for i in range(retry):
            try:
                print(f'LLM request 尝试第{i + 1}次...')
                response = self.llm_client.request(request_text)
                predecessors = get_predecessors_from_response(response, list(self.variables_description.keys()))
                break
            except Exception as e:
                print(e)
        if predecessors is None:
            raise Exception('LLM request 重试失败.')
        return predecessors


if __name__ == '__main__':
    v = {'a': 1, 'b': 2, 'c': 3}
    g = nx.DiGraph([('a', 'b'), ('a', 'c')])

    print(generate_request_text('123', 'abc', v,None, 'c', g))
