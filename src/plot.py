import matplotlib.pyplot as plt
from tqdm import tqdm
import metrics
import evaluation as eval
from torchscan import summary

HEADER_LENGTH = 80

# Method that prints the progress of the training/testing
def print_progress(title, batch_id, data, data_loader, loss, score=None):
    tqdm.write(
        "{}: [{}/{} ({:.0f}%)]\t Loss: {:.6f} {}".format(
            title,
            batch_id * len(data),
            len(data_loader.dataset),
            100.0 * batch_id / len(data_loader),
            loss.item(),
            "\tScore: " + str(score) if score is not None else "",
        )
    )


def print_header(title=""):
    if title == "":
        print("=" * HEADER_LENGTH)
    else:
        half_header = int((HEADER_LENGTH - len(title)) / 2)
        full_header = "=" * half_header + " " + title.upper() + " " + "=" * half_header
        print(full_header[:HEADER_LENGTH])


# Method that prints the average loss, optional metric score and the elapsed time
def print_performance(
    title, loss, duration, batch_duration, data_duration, metric=None, score=None
):
    print_header(title=title + " PERFORMANCE")
    print("Average loss = {:.4f}".format(loss))
    if metric is not None:
        if metric in metrics.NAMES.keys():
            print("{} = {:.4f}".format(metrics.NAMES[metric], score))
        else:
            print("Metric = {:.4f}".format(score))
    print(
        "Elapsed time = {:.2f} milliseconds ({:.2f} per batch, {:.2f} per data point)".format(
            duration, batch_duration, data_duration
        )
    )
    print_header()


# Method that plots the loss
def plot_loss(loss, it, it_per_epoch, smooth_loss=[], base_name="", title=""):
    fig = plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(loss)
    plt.plot(smooth_loss)
    epochs = [i * int(it_per_epoch) for i in range(int(it / it_per_epoch) + 1)]
    plt.plot(epochs, [loss[i] for i in epochs], linestyle="", marker="o")
    # if len(epochs) > 1: print(smooth_loss[epochs[-2]] -  smooth_loss[epochs[-1]] )
    plt.title(title)
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    # plt.ylim([0, 3])
    if base_name != "":
        fig.savefig(base_name + ".png")
        # pickle.dump([loss, smooth_loss, it], open(base_name + '-' + str(it) + '.p', 'wb'))
        # print(it)
    else:
        plt.show()
    plt.close("all")


# Method that prints the metrics if they are given
def print_metrics(**kwargs):
    print_header(title="METRICS")
    if "loss" in kwargs:
        print("Loss: {:.6f}".format(kwargs.get("loss")))
    if "score" in kwargs:
        print("Score: {:.6f}".format(kwargs.get("score")))
    if "batch_duration" in kwargs and "batch_size" in kwargs:
        print(
            "Time per batch: {:.4f} ms ({} per batch)".format(
                kwargs.get("batch_duration"), kwargs.get("batch_size")
            )
        )
    if "data_duration" in kwargs:
        print("Time per data point: {:.4f} ms".format(kwargs.get("data_duration")))
    if "model" in kwargs:
        model_size = eval.get_model_size(kwargs.get("model"))
        print("Model Size: {} MB".format(model_size))
        params = eval.get_model_parameters(kwargs.get("model"))
        print("Number of parameters: {}".format(params))
        if "example_input" in kwargs:
            model_flops, params = eval.get_model_flops(
                kwargs.get("model"), kwargs.get("example_input").size()
            )
            print("Number of FLOPS: {0}".format(model_flops))
    print_header()


# Method that prints a comparison of metrics
def print_before_after_metrics(before, after):
    # Helper function to check whether variable is in both sets
    in_both = lambda x: x in before and x in after
    print_header(title="METRICS BEFORE & AFTER")
    if in_both("loss"):
        print("Loss: {:.6f} -> {:.6f}".format(before.get("loss"), after.get("loss")))
    if in_both("score"):
        print(
            "Score: {:.6f} -> {:.6f} ".format(before.get("score"), after.get("score"))
        )
    if in_both("batch_duration") and in_both("batch_size"):
        assert before.get("batch_size") == after.get("batch_size")
        print(
            "Time per batch: {:.4f} ms -> {:.4f} ms ({} per batch)".format(
                before.get("batch_duration"),
                after.get("batch_duration"),
                before.get("batch_size"),
            )
        )
    if in_both("data_duration"):
        print(
            "Time per data point: {:.4f} ms -> {:.4f} ms".format(
                before.get("data_duration"), after.get("data_duration")
            )
        )
    if in_both("model"):
        before_model_size = eval.get_model_size(before.get("model"))
        after_model_size = eval.get_model_size(after.get("model"))
        print("Model Size: {} MB -> {} MB".format(before_model_size, after_model_size))
        before_params = eval.get_model_parameters(before.get("model"))
        after_params = eval.get_model_parameters(after.get("model"))
        print("Number of parameters: {} -> {}".format(before_params, after_params))
        if in_both("example_input"):
            before_flops, _ = eval.get_model_flops(
                before.get("model"), before.get("example_input").size()
            )
            after_flops, _ = eval.get_model_flops(
                after.get("model"), after.get("example_input").size()
            )
            print("Number of FLOPS: {} -> {}".format(before_flops, after_flops))
    print_header()


def get_summary(model, example_input):
    summary(model, example_input.size())
