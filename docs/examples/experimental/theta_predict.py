import numpy as np
from thetan.berserker import random_walk_auto
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
# import joblib 

def gaussian_clusters(n_local_samples, blob_dis, cluster_x=30, cluster_y=10):
    xx = np.arange(-blob_dis*cluster_x//2, blob_dis*cluster_x//2, blob_dis)
    yy = np.arange(-blob_dis*cluster_y//2, blob_dis*cluster_y//2, blob_dis)
    xx, yy = np.meshgrid(xx, yy)
    centres = np.column_stack((xx.ravel(), yy.ravel()))

    X = np.zeros((n_local_samples * len(centres), 2))
    for i, cpos in enumerate(centres):
        pts = cpos + np.random.normal(0, 1, size=(n_local_samples, 2))
        X[i*n_local_samples:(i+1)*n_local_samples] = pts
    return X

def build_histograms(inter_dists, n_hist_per_theta=50, reps=100, bins=100):
    hists = []
    thetas = []
    for theta in inter_dists:
        print(f"→ Generating {n_hist_per_theta} histograms for θ = {theta}")
        for _ in range(n_hist_per_theta):
            X = gaussian_clusters(500, blob_dis=theta)
            dists = np.asarray(random_walk_auto(X, reps=reps))
            hist, _ = np.histogram(dists.ravel(), bins=bins)
            hists.append(hist)
            thetas.append(theta)
    return np.stack(hists), np.array(thetas, dtype=float)

def main():
    inter_dists      = [5, 15, 25, 35, 45]
    n_hist_per_theta = 50
    random_walk_reps = 100
    n_bins           = 100

    X_feat, y_tar = build_histograms(
        inter_dists,
        n_hist_per_theta=n_hist_per_theta,
        reps=random_walk_reps,
        bins=n_bins
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y_tar, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\nEvaluation on hold-out set:")
    print(f"  R² score: {r2:.3f}")
    print(f"  MAE:      {mae:.3f}\n")

    print("Actual θ  →  Predicted θ")
    print("------------------------")
    for actual, pred in zip(y_test, y_pred):
        print(f"   {actual:5.1f}   →    {pred:5.2f}")

  
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], linestyle='--', color='gray')
    plt.xlabel("True θ")
    plt.ylabel("Predicted θ")
    plt.title("Linear Regression: True vs. Predicted θ")
    plt.tight_layout()
    plt.show()

    # Save the model if required
    # joblib.dump(model, "theta_linear_regressor.joblib")
    # print("\n✅ Model saved as 'theta_linear_regressor.joblib'")

if __name__ == "__main__":
    main()
