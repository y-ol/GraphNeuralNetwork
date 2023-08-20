import click
import gridS as g 
import eval as e 

DATASETS  = e.datasets


@click.command()
@click.option("--dataset", prompt="Enter dataset ", type=click.Choice(DATASETS))
@click.option("--epochs", prompt="Number of epochs ", type = click.IntRange(min = 1), default = 200)
@click.option("--num-repeats", prompt= "Number of repeats", type = click.IntRange(min = 1), default = 3)
@click.option("--num-layers",  type=click.Choice(g.param_grid["num_layers"]), multiple=True, default = None)
@click.option("--learning-rate",  type=click.Choice(g.param_grid["learning_rate"]), multiple=True, default=None)
@click.option("--regularization",  type=click.Choice(g.param_grid["regularization"]), multiple=True, default=None)
@click.option("--probability",  type=click.Choice(g.param_grid["probability"]), multiple=True, default=None)
@click.option("--activation",  type=click.Choice(g.param_grid["activation"]), multiple=True, default=None)
@click.option("--units",  type=click.Choice(g.param_grid["units"]), multiple=True, default=None)
@click.option("--convo-type",  type=click.Choice(g.param_grid["convo_type"]), multiple=True, default=None)


def run(dataset, epochs, num_repeats, num_layers, learning_rate, regularization, probability, activation, units, convo_type): 
    g.train_and_evaluate(hyperparams=g.param_combos, dataset_name=dataset, epochs = epochs, 
                         experiment_results_dir='/home/olga/GraphNeuralNetwork', num_repeats=num_repeats, 
                         filter=e.filter_builder(num_layers=num_layers, learning_rate=learning_rate, regularization=regularization, probability=probability, activation=activation, units=units, convo_type=convo_type))

if __name__ == "__main__":
    run()
    



# @click.command()
# @click.option('--count', default=1, help='Number of greetings.')
# @click.option('--name', prompt='Your name',
#               help='The person to greet.')
# def hello(count, name):
#     """Simple program that greets NAME for a total of COUNT times."""
#     for x in range(count):
#         click.echo(f"Hello {name}!")


# if __name__ == '__main__':
#     hello()

