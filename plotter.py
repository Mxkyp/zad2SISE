import numpy as np
import matplotlib.pyplot as plt


def plot_error_distributions(
    Y_test_original, Y_test_corrected, Y_test_uncorrected, title_suffix=""
):
    """Porównanie rozkładów błędów przed i po korekcji"""

    # Obliczenie błędów absolutnych
    errors_original = np.sqrt(Y_test_original[:, 0] ** 2 + Y_test_original[:, 1] ** 2)
    errors_corrected = np.sqrt(
        Y_test_corrected[:, 0] ** 2 + Y_test_corrected[:, 1] ** 2
    )

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Histogram błędów
    ax1.hist(errors_original, bins=50, alpha=0.7, label="Przed korekcją", density=True)
    ax1.hist(errors_corrected, bins=50, alpha=0.7, label="Po korekcji", density=True)
    ax1.set_xlabel("Błąd [m]")
    ax1.set_ylabel("Gęstość")
    ax1.set_title(f"Rozkład błędów {title_suffix}")
    ax1.legend()
    ax1.grid(True)

    # Dystrybuanta CDF
    sorted_orig = np.sort(errors_original)
    sorted_corr = np.sort(errors_corrected)
    cdf_orig = np.arange(1, len(sorted_orig) + 1) / len(sorted_orig)
    cdf_corr = np.arange(1, len(sorted_corr) + 1) / len(sorted_corr)

    ax2.plot(sorted_orig, cdf_orig, label="Przed korekcją", linewidth=2)
    ax2.plot(sorted_corr, cdf_corr, label="Po korekcji", linewidth=2)
    ax2.set_xlabel("Błąd [m]")
    ax2.set_ylabel("Dystrybuanta F(x)")
    ax2.set_title(f"Dystrybuanta błędów {title_suffix}")
    ax2.legend()
    ax2.grid(True)

    # Scatter plot błędów X vs Y (dodane: uncorrected)
    ax3.scatter(
        Y_test_uncorrected[:, 0],
        Y_test_uncorrected[:, 1],
        alpha=0.5,
        label="Przed korekcją modelu",
        s=1,
        color="gray",
    )
    ax3.scatter(
        Y_test_original[:, 0],
        Y_test_original[:, 1],
        alpha=0.5,
        label="Przed korekcją końcową",
        s=1,
        color="blue",
    )
    ax3.scatter(
        Y_test_corrected[:, 0],
        Y_test_corrected[:, 1],
        alpha=0.5,
        label="Po korekcji",
        s=1,
        color="orange",
    )
    ax3.set_xlabel("Błąd X [m]")
    ax3.set_ylabel("Błąd Y [m]")
    ax3.set_title(f"Błędy X vs Y {title_suffix}")
    ax3.legend()
    ax3.grid(True)
    ax3.axis("equal")

    plt.tight_layout()
    plt.savefig(
        f"error_analysis_{title_suffix.replace(' ', '_')}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    return sorted_corr, cdf_corr
