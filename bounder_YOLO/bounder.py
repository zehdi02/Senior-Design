from ultralytics import YOLO
import matplotlib.pyplot as plt


def main():
    model = YOLO("yolov8n.pt")

    result_grid = model.train(
        data='manga109.yaml',
        epochs=10,
        batch=16,
        device='cuda',
        dropout=.05,
        val=True,
        save=True,
        resume=True,
        cache=True
    )

    """
    for i, result in enumerate(result_grid):
        plt.plot(
            result.metrics_dataframe["training_iteration"],
            result.metrics_dataframe["mean_accuracy"],
            label=f"Trial {i}",
        )

    plt.xlabel("Training Iterations")
    plt.ylabel("Mean Accuracy")
    plt.legend()
    plt.show()
    """


if __name__ == '__main__':
    main()


"""
"""