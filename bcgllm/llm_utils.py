import time
from copy import copy


def get_possible_actions_with_llm(client, dag, possibleActionsInfo, domain=None,
                                  background=None, var_info=None,
                                  num_limit=None, one_step_num_limit=None, retry=5):
    if one_step_num_limit is not None:
        sum_len = len(possibleActionsInfo)
        if sum_len > one_step_num_limit:
            return get_possible_actions_with_llm_multi_step(client=client, dag=dag,
                                                            possibleActionsInfo=possibleActionsInfo,
                                                            domain=domain, background=background, var_info=var_info,
                                                            num_limit=num_limit, one_step_num_limit=one_step_num_limit,
                                                            retry=retry)
    return get_possible_actions_with_llm_one_step(client=client, dag=dag, possibleActionsInfo=possibleActionsInfo,
                                                  domain=domain,
                                                  background=background, var_info=var_info,
                                                  num_limit=num_limit,
                                                  retry=retry)


def get_possible_actions_with_llm_one_step(client, dag, possibleActionsInfo,
                                           domain=None, background=None, var_info=None,
                                           num_limit=None, sort=True, retry=5):
    request_text = generate_possible_actions_text(dag, possibleActionsInfo=possibleActionsInfo,
                                                  domain=domain,
                                                  background=background,
                                                  var_info=var_info,
                                                  num_limit=num_limit, sort=sort)
    possible_actions = None
    t_s = time.time()
    for i in range(retry):
        try:
            print(f'LLM request try {i + 1} ...')
            response_text = client.request(request_text)
            answer_list = response_text_to_list(response_text)
            possible_actions = []
            for actions_dict_index in answer_list:
                possible_actions.append(possibleActionsInfo[actions_dict_index])
            break
        except Exception as e:
            print(e)
    print('time_choice_child_node_with_llm:', time.time() - t_s)
    if possible_actions is None:
        print('choice_child_node_with_llm failed.')
    return possible_actions


def get_possible_actions_with_llm_multi_step(client, dag, possibleActionsInfo,
                                             domain=None, background=None, var_info=None,
                                             num_limit=None, one_step_num_limit=None, retry=5):
    all_actions = copy(possibleActionsInfo)

    patience = 0
    patience_limit = 4
    while len(all_actions) > one_step_num_limit and patience < patience_limit:
        patience += 1
        new_all_actions = []
        while all_actions:
            if len(all_actions) >= one_step_num_limit:
                temp_actions = all_actions[:one_step_num_limit]
                all_actions = all_actions[one_step_num_limit:]
            else:
                temp_actions = all_actions.copy()
                all_actions = []
            try:
                new_all_actions.extend(
                    get_possible_actions_with_llm_one_step(
                        client=client, dag=dag, possibleActionsInfo=temp_actions,
                        domain=domain, background=background, var_info=var_info,
                        sort=False, retry=retry))
            except Exception as e:
                print(e)
                new_all_actions.extend(temp_actions)
        all_actions = new_all_actions

    return get_possible_actions_with_llm_one_step(
        client=client, dag=dag, possibleActionsInfo=all_actions,
        domain=domain, background=background, var_info=var_info,
        num_limit=num_limit, sort=True, retry=retry)


# def actions_to_three_list(actions):
#     add_edges = []
#     remove_edges = []
#     reverse_edges = []
#     for ops, u, v in actions:
#         if ops == 'add':
#             add_edges.append((u, v))
#         elif ops == 'remove':
#             remove_edges.append((u, v))
#         elif ops == 'reverse':
#             reverse_edges.append((u, v))
#     return add_edges, remove_edges, reverse_edges
#
#
# def three_list_to_actions(add_edges, remove_edges, reverse_edges):
#     actions = []
#     for u, v in add_edges:
#         actions.append(('add', u, v))
#     for u, v in remove_edges:
#         actions.append(('remove', u, v))
#     for u, v in reverse_edges:
#         actions.append(('reverse', u, v))
#     return actions


def generate_possible_actions_text(dag, possibleActionsInfo, domain=None,
                                   background=None, var_info=None, num_limit=None, sort=True):
    start_text = ''
    background_text = ''
    current_state_text = ''
    candidate_text = ''
    end_text = ''

    start_text = f'You are an expert in the fields of {domain} and causal discovery.'

    background_text_list = []
    if background:
        background_text_list.append('Background:')
        background_text_list.append(background)
    if var_info:
        background_text_list.append(
            'There are the following variables:')
        var_info_text_list = []
        for var, info in var_info.items():
            if info:
                var_info_text_list.append(f'{var}: {info}')
            else:
                var_info_text_list.append(f'{var}')
        var_info_text = ',\n'.join(var_info_text_list)
        background_text_list.append(var_info_text)
    background_text_list.append(
        'Now I have a conjectured causal graph, please use your knowledge to help me improve it.')
    background_text = '\n\n'.join(background_text_list)

    candidate_text_list = []
    candidate_text_list.append(
        'The following are possible modifications to the causal graph:')
    index = 0
    for actionsInfo in possibleActionsInfo:
        index += 1
        ops = actionsInfo['ops']
        u = actionsInfo['u']
        v = actionsInfo['v']
        if ops == 'add':
            condition_vars = actionsInfo['condition_vars']
            p_value = actionsInfo['p_value']
            one_action_text_list = []
            one_action_text_list.append(f'{index}. add edge {u} -> {v}. {u} directly causes {v}.')
            if condition_vars:
                one_action_text_list.append(
                    f'Before adding edge, the causal graph implies that {u} should be independent of {v} given {condition_vars}.')
            else:
                one_action_text_list.append(
                    f'Before adding edge, the causal graph implies that {u} should be independent of {v}.')
            one_action_text_list[-1] += f'conditional independence test p-value: {p_value:0.3}'
            one_action_text = '\n'.join(one_action_text_list)
            candidate_text_list.append(one_action_text)
        elif ops == 'remove':
            condition_vars = actionsInfo['condition_vars']
            p_value = actionsInfo['p_value']
            one_action_text_list = []
            one_action_text_list.append(f'{index}. remove edge {u} -> {v}. {u} does not directly causes {v}.')
            if condition_vars:
                one_action_text_list.append(
                    f'After deleting edge, the causal graph implies that {u} should be independent of {v} given {condition_vars}.')
            else:
                one_action_text_list.append(
                    f'After deleting edge, the causal graph implies that {u} should be independent of {v}.')
            one_action_text_list[-1] += f'conditional independence test p-value: {p_value:0.3}'
            one_action_text = '\n'.join(one_action_text_list)
            candidate_text_list.append(one_action_text)
        elif ops == 'reverse':
            condition_vars_before = actionsInfo['condition_vars_before']
            p_value_before = actionsInfo['p_value_before']
            condition_vars_after = actionsInfo['condition_vars_after']
            p_value_after = actionsInfo['p_value_after']
            one_action_text_list = []
            one_action_text_list.append(
                f'{index}. reverse edge {u} -> {v}. {v} directly causes {u}, rather than the original speculation that {u} directly causes {v}.')
            if condition_vars_after:
                one_action_text_list.append(
                    f'After reversing edge, the causal graph implies that {u} should be dependent of {v} given {condition_vars_after}.')
            else:
                one_action_text_list.append(
                    f'After reversing edge, the causal graph implies that {u} should be dependent of {v}.')
            one_action_text_list[-1] += f'conditional independence test p-value: {p_value_after:0.3}'
            one_action_text = '\n'.join(one_action_text_list)
            candidate_text_list.append(one_action_text)
    candidate_text = '\n\n'.join(candidate_text_list)

    end_text_list = []
    end_text_list.append(
        f'Please evaluate the correctness of the above modifications based on your expert knowledge and in-depth understanding of the {domain} and causal discovery field and conditional independence test result.')
    if num_limit is None:
        end_text_list.append(
            'Please select the options with a higher likelihood of being correct')
    else:
        end_text_list.append(
            f'Please select the options(no more than {num_limit} options) with a higher likelihood of being correct')
    if sort:
        end_text_list[-1] += ' and sort them according to the correct likelihood.'
    else:
        end_text_list[-1] += '.'
    end_text_list.append(
        'Let\'s work this out in a step by step way to be sure that we have the right answer. Then provide your final answer(only the number for options, separated by \',\') with in the tags <Answer> </Answer>.')
    end_text = '\n'.join(end_text_list)

    question_text = '\n\n'.join([start_text, background_text, candidate_text, end_text])
    return question_text


def response_text_to_list(response_text):
    start_index = response_text.index('<Answer>') + 8
    end_index = response_text.index('</Answer>')
    text_list = response_text[start_index:end_index].split(',')
    answer_list = []
    for text in text_list:
        try:
            answer_list.append(int(text) - 1)
        except Exception as e:
            print('ERROR: response_text_to_list:', e)
    return answer_list
