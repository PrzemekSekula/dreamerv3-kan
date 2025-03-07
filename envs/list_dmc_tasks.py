from dm_control import suite

# Iterate through the tuple of tuples
for domain_name, task in suite.ALL_TASKS:
    print(f"{domain_name} task: {task}")
