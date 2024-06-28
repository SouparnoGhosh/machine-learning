def normalize_data(X, data):
    #* Z-Score Normalization
    return (X - data.mean()) / data.std()




def print_accuracy_table(optimizers, optimizer_metrics):
    print("\nAccuracy Table:")
    print(f'+{"-" * 22}+{"-" * 22}+{"-" * 27}+{"-" * 22}+')
    print(f'| {"Optimizer":<20} | {"Max Accuracy":<20} | {"Epoch of Max Accuracy":<25} | {"Last Epoch Accuracy":<20} |')
    print(f'+{"-" * 22}+{"-" * 22}+{"-" * 27}+{"-" * 22}+')

    for optimizer_name in optimizers.keys():
        accuracies = optimizer_metrics[optimizer_name]["accuracy"]
        max_accuracy = max(accuracies)
        epoch_max_accuracy = accuracies.index(max_accuracy)
        last_epoch_accuracy = accuracies[-1]
        print(f'| {optimizer_name:<20} | {max_accuracy:<20.2f} | {epoch_max_accuracy:<25} | {last_epoch_accuracy:<20.2f} |')

    print(f'+{"-" * 22}+{"-" * 22}+{"-" * 27}+{"-" * 22}+')


def print_loss_table(optimizers, optimizer_metrics):
    print("\nLoss Table:")
    print(f'+{"-" * 22}+{"-" * 22}+{"-" * 27}+{"-" * 22}+')
    print(f'| {"Optimizer":<20} | {"Min Loss":<20} | {"Epoch of Min Loss":<25} | {"Last Epoch Loss":<20} |')
    print(f'+{"-" * 22}+{"-" * 22}+{"-" * 27}+{"-" * 22}+')

    for optimizer_name in optimizers.keys():
        losses = optimizer_metrics[optimizer_name]["loss"]
        min_loss = min(losses)
        epoch_min_loss = losses.index(min_loss)
        last_epoch_loss = losses[-1]
        print(f'| {optimizer_name:<20} | {min_loss:<20.2f} | {epoch_min_loss:<25} | {last_epoch_loss:<20.2f} |')

    print(f'+{"-" * 22}+{"-" * 22}+{"-" * 27}+{"-" * 22}+')

def print_tables(optimizer, optimizer_metrics):
    print_accuracy_table(optimizer, optimizer_metrics)
    print_loss_table(optimizer, optimizer_metrics)
