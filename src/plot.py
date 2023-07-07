import matplotlib.pyplot as plt
from tqdm import tqdm
import metrics

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
def print_results(**kwargs):
    print_header(title="RESULTS")
    if "loss" in kwargs:
        print("Loss: {:.6f}".format(kwargs["loss"]))
    if "score" in kwargs:
        print("Score: {:.6f}".format(kwargs["score"]))
    if "batch_duration" in kwargs and "batch_size" in kwargs:
        print(
            "Time per batch: {:.4f} ms ({} per batch)".format(
                kwargs["batch_duration"], kwargs["batch_size"]
            )
        )
    if "data_duration" in kwargs:
        print("Time per data point: {:.4f} ms".format(kwargs["data_duration"]))
    if "model_size" in kwargs:
        print("Model Size: {} MB".format(kwargs["model_size"]))
    if "params" in kwargs:
        print("Number of parameters: {}".format(kwargs["params"]))
    if "flops" in kwargs:
        print("Number of FLOPs: {0}".format(kwargs["flops"]))
    if "macs" in kwargs:
        print("Number of MACs: {0}".format(kwargs["macs"]))
    print_header()


# Method that prints a comparison of metrics
def print_before_after_results(before, after):
    # Helper function to check whether variable is in both sets
    in_both = lambda x: x in before and x in after
    reduction = lambda x: -1 * (1 - (after[x] / before[x])) * 100
    print_header(title="RESULTS BEFORE & AFTER")
    if in_both("loss"):
        print("Loss: {:.6f} -> {:.6f} ({:.2f}%)".format(before["loss"], after["loss"], reduction("loss")))
    if in_both("score"):
        print("Score: {:.6f} -> {:.6f} ({:.2f}%)".format(before["score"], after["score"], reduction("score")))
    if in_both("batch_duration") and in_both("batch_size"):
        assert before["batch_size"] == after["batch_size"]
        print(
            "Time per batch: {:.4f} ms -> {:.4f} ms ({:.2f}%) ({} per batch)".format(
                before["batch_duration"],
                after["batch_duration"],
                reduction("batch_duration"),
                before["batch_size"],
            )
        )
    if in_both("data_duration"):
        print(
            "Time per data point: {:.4f} ms -> {:.4f} ms ({:.2f}%)".format(
                before["data_duration"], after["data_duration"], reduction("data_duration")
            )
        )
    if in_both("model_size"):
        print(
            "Model Size: {} MB -> {} MB ({:.2f}%)".format(
                before["model_size"], after["model_size"], reduction("model_size")
            ))
    if in_both("params"):
        print(
            "Number of parameters: {} -> {} ({:.2f}%)".format(before["params"], after["params"], reduction("params")))
    if in_both("flops"):
        print("Number of FLOPs: {} -> {} ({:.2f}%)".format(before["flops"], after["flops"], reduction("flops")))
    if in_both("macs"):
        print("Number of MACs: {} -> {} ({:.2f}%)".format(before["macs"], after["macs"], reduction("macs")))
    print_header()


# Method that prints a header
def log_metrics_to_tensorboard(writer, phase, train_metrics, val_metrics, step):
    train_loss, train_score, train_duration, train_batch_duration, train_data_duration = train_metrics
    val_loss, val_score, val_duration, val_batch_duration, val_data_duration = val_metrics

    writer.add_scalars(f"loss/{phase}", {"train": train_loss, "validation": val_loss}, step)
    writer.add_scalars(f"score/{phase}", {"train": train_score, "validation": val_score}, step)
    writer.add_scalars(f"batch duration/{phase}", {"train": train_batch_duration, "validation": val_batch_duration}, step)
    writer.add_scalars(f"data duration/{phase}", {"train": train_data_duration, "validation": val_data_duration}, step)
    writer.flush()

# General metrics logging method
def log_metrics(writer, phase, label, metrics, step):
     for metric in metrics:
        writer.add_scalars(f"{metric}/{phase}", {label: metrics[metric]}, step)
        writer.flush()