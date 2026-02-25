import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc,
    RocCurveDisplay,
)


# ---------------------------
# Config Streamlit
# ---------------------------
st.set_page_config(
    page_title="Iris Classifier Lab",
    layout="wide",
)


# ---------------------------
# Datos y utilidades
# ---------------------------
@st.cache_data
def load_data() -> Tuple[pd.DataFrame, pd.Series, Dict[int, str]]:
    iris = load_iris(as_frame=True)
    X = iris.data.copy()
    y = iris.target.copy()
    names = {i: name for i, name in enumerate(iris.target_names)}
    return X, y, names


@dataclass
class ModelSpec:
    name: str
    estimator: Any
    supports_proba: bool


def get_model_specs(needs_proba: bool, params: Dict[str, Any]) -> Dict[str, ModelSpec]:
    """
    Devuelve un dict de modelos configurados.
    Si necesitas ROC, algunos modelos requieren probability=True (SVM).
    """
    # Logistic Regression
    lr = LogisticRegression(
        C=params.get("lr_C", 1.0),
        max_iter=params.get("lr_max_iter", 1000),
        solver="lbfgs",
        multi_class="auto",
    )

    # SVM
    svm_probability = True if needs_proba else params.get("svm_probability", False)
    svm = SVC(
        C=params.get("svm_C", 1.0),
        kernel=params.get("svm_kernel", "rbf"),
        gamma=params.get("svm_gamma", "scale"),
        probability=svm_probability,
    )

    # KNN
    knn = KNeighborsClassifier(
        n_neighbors=params.get("knn_k", 5),
        weights=params.get("knn_weights", "uniform"),
    )

    # Decision Tree
    dt = DecisionTreeClassifier(
        max_depth=params.get("dt_max_depth", None),
        min_samples_split=params.get("dt_min_samples_split", 2),
        random_state=params.get("random_state", 42),
    )

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=params.get("rf_n_estimators", 200),
        max_depth=params.get("rf_max_depth", None),
        random_state=params.get("random_state", 42),
    )

    # Naive Bayes
    nb = GaussianNB()

    # LDA
    lda = LinearDiscriminantAnalysis()

    specs = {
        "Logistic Regression": ModelSpec("Logistic Regression", lr, supports_proba=True),
        "SVM": ModelSpec("SVM", svm, supports_proba=svm_probability),
        "KNN": ModelSpec("KNN", knn, supports_proba=True),  # KNN sí tiene predict_proba
        "Decision Tree": ModelSpec("Decision Tree", dt, supports_proba=True),
        "Random Forest": ModelSpec("Random Forest", rf, supports_proba=True),
        "Naive Bayes (Gaussian)": ModelSpec("Naive Bayes (Gaussian)", nb, supports_proba=True),
        "LDA": ModelSpec("LDA", lda, supports_proba=True),
    }
    return specs


def make_pipeline(estimator, use_scaler: bool) -> Pipeline:
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def make_2d_representation(
    X: pd.DataFrame,
    mode_2d: str,
    two_features: Tuple[str, str],
    use_scaler_for_2d: bool,
    random_state: int,
) -> Tuple[np.ndarray, Optional[Pipeline], str]:
    """
    Retorna X2 (n,2) y un transformador (opcional) para el modo PCA.
    - mode_2d = "Features" o "PCA"
    """
    if mode_2d == "PCA (2 componentes)":
        # Pipeline para estandarizar y PCA a 2D
        steps = []
        if use_scaler_for_2d:
            steps.append(("scaler", StandardScaler()))
        steps.append(("pca", PCA(n_components=2, random_state=random_state)))
        transformer = Pipeline(steps)
        X2 = transformer.fit_transform(X.values)
        xlab, ylab = "PC1", "PC2"
        return X2, transformer, f"{xlab} vs {ylab}"
    else:
        f1, f2 = two_features
        X2 = X[[f1, f2]].values
        return X2, None, f"{f1} vs {f2}"


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> Dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


def plot_confusion(y_true, y_pred, class_names) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de Confusión")
    fig.tight_layout()
    return fig


def plot_multiclass_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: list,
) -> plt.Figure:
    """
    y_score: (n_samples, n_classes) con probabilidades o scores.
    """
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots()

    # ROC por clase
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.3f})")

    # Micro-average
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_score.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro, linestyle="--", label=f"Micro-average (AUC={roc_auc_micro:.3f})")

    ax.plot([0, 1], [0, 1], linestyle=":", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Multiclase (One-vs-Rest)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_decision_boundary_2d(
    X2_train: np.ndarray,
    y_train: np.ndarray,
    X2_test: np.ndarray,
    y_test: np.ndarray,
    model_2d: Pipeline,
    class_names: list,
    title: str,
    mesh_step: float = 0.02,
) -> plt.Figure:
    # Rango de malla
    x_min, x_max = X2_train[:, 0].min() - 0.6, X2_train[:, 0].max() + 0.6
    y_min, y_max = X2_train[:, 1].min() - 0.6, X2_train[:, 1].max() + 0.6
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step), np.arange(y_min, y_max, mesh_step))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predicción sobre la malla
    Z = model_2d.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.25)

    # puntos train y test
    scatter_train = ax.scatter(X2_train[:, 0], X2_train[:, 1], c=y_train, marker="o", edgecolors="k", label="Train")
    scatter_test = ax.scatter(X2_test[:, 0], X2_test[:, 1], c=y_test, marker="^", edgecolors="k", label="Test")

    # Leyenda de clases (manual)
    handles = []
    for i, cname in enumerate(class_names):
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="", label=cname))
    ax.legend(handles=handles + [scatter_train, scatter_test], loc="best")

    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    fig.tight_layout()
    return fig


# ---------------------------
# UI
# ---------------------------
X, y, name_map = load_data()
class_names = [name_map[i] for i in sorted(name_map.keys())]

st.title("🌸 Iris Dataset - Clasificación con Streamlit")
st.write(
    "App interactiva para entrenar **modelos de clasificación**, comparar **métricas** y visualizar "
    "**frontera de decisión (2D)** y **curvas ROC**."
)

with st.expander("📌 Ver dataset (preview)"):
    st.dataframe(pd.concat([X, y.rename("target")], axis=1).head(10), use_container_width=True)
    st.caption(f"Filas: {len(X)} | Features: {list(X.columns)} | Clases: {class_names}")

# Sidebar: configuración general
st.sidebar.header("⚙️ Configuración")

random_state = st.sidebar.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)
test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.5, value=0.25, step=0.05)
use_scaler = st.sidebar.checkbox("Estandarizar (StandardScaler)", value=True)

train_mode = st.sidebar.radio(
    "Modo de entrenamiento (métricas)",
    ["4 features (mejor desempeño)", "2D (para frontera)"],
    index=0,
)

st.sidebar.divider()
st.sidebar.subheader("📈 Visualización 2D (frontera)")

mode_2d = st.sidebar.selectbox("Representación 2D", ["Features (2 columnas)", "PCA (2 componentes)"])

two_features = st.sidebar.multiselect(
    "Elige 2 features (si usas Features 2D)",
    options=list(X.columns),
    default=[X.columns[0], X.columns[2]],
    max_selections=2,
)

use_scaler_for_2d = st.sidebar.checkbox("Estandarizar antes de PCA/2D", value=True)

st.sidebar.divider()
st.sidebar.subheader("🤖 Modelo y métricas")

show_roc = st.sidebar.checkbox("Mostrar ROC (multiclase)", value=True)
show_cv = st.sidebar.checkbox("Mostrar validación cruzada (5-fold)", value=False)

metric_average = st.sidebar.selectbox("Promedio para Precision/Recall/F1", ["macro", "weighted"], index=0)

# Parámetros del modelo
st.sidebar.divider()
st.sidebar.subheader("🔧 Hiperparámetros")

params: Dict[str, Any] = {"random_state": random_state}

model_choice = st.sidebar.selectbox(
    "Selecciona un modelo",
    [
        "Logistic Regression",
        "SVM",
        "KNN",
        "Decision Tree",
        "Random Forest",
        "Naive Bayes (Gaussian)",
        "LDA",
    ],
)

if model_choice == "Logistic Regression":
    params["lr_C"] = st.sidebar.slider("C", 0.01, 10.0, 1.0, 0.01)
    params["lr_max_iter"] = st.sidebar.slider("max_iter", 200, 5000, 1000, 100)

elif model_choice == "SVM":
    params["svm_C"] = st.sidebar.slider("C", 0.01, 10.0, 1.0, 0.01)
    params["svm_kernel"] = st.sidebar.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"], index=0)
    params["svm_gamma"] = st.sidebar.selectbox("gamma", ["scale", "auto"], index=0)
    params["svm_probability"] = st.sidebar.checkbox("probability=True (más lento)", value=show_roc)

elif model_choice == "KNN":
    params["knn_k"] = st.sidebar.slider("n_neighbors", 1, 30, 5, 1)
    params["knn_weights"] = st.sidebar.selectbox("weights", ["uniform", "distance"], index=0)

elif model_choice == "Decision Tree":
    md = st.sidebar.slider("max_depth (0 = None)", 0, 10, 0, 1)
    params["dt_max_depth"] = None if md == 0 else md
    params["dt_min_samples_split"] = st.sidebar.slider("min_samples_split", 2, 10, 2, 1)

elif model_choice == "Random Forest":
    params["rf_n_estimators"] = st.sidebar.slider("n_estimators", 50, 500, 200, 50)
    md = st.sidebar.slider("max_depth (0 = None)", 0, 20, 0, 1)
    params["rf_max_depth"] = None if md == 0 else md


# ---------------------------
# Preparar datasets
# ---------------------------
# Verificación simple para 2 features cuando aplique
if mode_2d == "Features (2 columnas)" and len(two_features) != 2:
    st.warning("Selecciona exactamente 2 features para la visualización 2D.")
    st.stop()

# Split principal para métricas
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=float(test_size),
    random_state=int(random_state),
    stratify=y,
)

# 2D data (para frontera)
mode_2d_internal = "PCA (2 componentes)" if mode_2d == "PCA (2 componentes)" else "Features"
two_feats_tuple = (two_features[0], two_features[1]) if len(two_features) == 2 else (X.columns[0], X.columns[1])

X2_all, transformer_2d, rep_title = make_2d_representation(
    X=X,
    mode_2d=("PCA (2 componentes)" if mode_2d == "PCA (2 componentes)" else "Features"),
    two_features=two_feats_tuple,
    use_scaler_for_2d=use_scaler_for_2d,
    random_state=int(random_state),
)

# Split 2D consistente con split principal (usamos índices)
train_idx = X_train.index.to_numpy()
test_idx = X_test.index.to_numpy()
X2_train = X2_all[train_idx]
X2_test = X2_all[test_idx]


# ---------------------------
# Entrenar modelos
# ---------------------------
# ¿Necesitamos proba? (para ROC)
needs_proba = bool(show_roc)

specs = get_model_specs(needs_proba=needs_proba, params=params)
spec = specs[model_choice]

# Modelo para métricas
if train_mode == "4 features (mejor desempeño)":
    pipeline_main = make_pipeline(spec.estimator, use_scaler=use_scaler)
    pipeline_main.fit(X_train, y_train)
    y_pred = pipeline_main.predict(X_test)

    y_score = None
    if show_roc and hasattr(pipeline_main, "predict_proba"):
        try:
            y_score = pipeline_main.predict_proba(X_test)
        except Exception:
            y_score = None

else:
    # Entrena con 2D (frontera y métricas coinciden)
    pipeline_main = make_pipeline(spec.estimator, use_scaler=use_scaler)
    pipeline_main.fit(X2_train, y_train.to_numpy())
    y_pred = pipeline_main.predict(X2_test)

    y_score = None
    if show_roc and hasattr(pipeline_main, "predict_proba"):
        try:
            y_score = pipeline_main.predict_proba(X2_test)
        except Exception:
            y_score = None

# Modelo 2D exclusivo para frontera si entrenas con 4 features
pipeline_2d = None
if train_mode == "4 features (mejor desempeño)":
    pipeline_2d = make_pipeline(spec.estimator, use_scaler=use_scaler)
    pipeline_2d.fit(X2_train, y_train.to_numpy())
else:
    pipeline_2d = pipeline_main


# ---------------------------
# Tabs de salida
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📌 Resumen", "📊 Métricas", "📈 Curvas", "🧠 Frontera de decisión"])

with tab1:
    st.subheader("Resumen del experimento")
    colA, colB, colC = st.columns(3)

    with colA:
        st.metric("Modelo", spec.name)
        st.metric("Modo métricas", train_mode)

    with colB:
        st.metric("Test size", f"{test_size:.2f}")
        st.metric("Scaler", "Sí" if use_scaler else "No")

    with colC:
        st.metric("Visualización 2D", mode_2d)
        st.metric("Promedio métricas", metric_average)

    if train_mode == "4 features (mejor desempeño)":
        st.info(
            "Nota: Estás entrenando el modelo principal con **4 features** para mejores métricas. "
            "La **frontera de decisión** se calcula con un **modelo 2D** separado (solo para visualizar)."
        )
    else:
        st.success("En modo 2D, las métricas y la frontera corresponden al mismo modelo y representación.")

    st.write("**Features del dataset:**", list(X.columns))
    if mode_2d == "Features (2 columnas)":
        st.write("**2D usando:**", list(two_feats_tuple))
    else:
        st.write("**2D usando:** PCA (2 componentes)")

with tab2:
    st.subheader("Métricas de desempeño")

    # Métricas
    y_true_np = y_test.to_numpy() if train_mode == "4 features (mejor desempeño)" else y_test.to_numpy()
    y_pred_np = np.asarray(y_pred)

    metrics = compute_metrics(y_true_np, y_pred_np, average=metric_average)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
    m2.metric("Precision", f"{metrics['Precision']:.3f}")
    m3.metric("Recall", f"{metrics['Recall']:.3f}")
    m4.metric("F1", f"{metrics['F1']:.3f}")

    # Confusion matrix
    fig_cm = plot_confusion(y_true_np, y_pred_np, class_names)
    st.pyplot(fig_cm, use_container_width=True)

    # Classification report
    with st.expander("📄 Ver classification_report"):
        rep = classification_report(y_true_np, y_pred_np, target_names=class_names, zero_division=0)
        st.text(rep)

    # Cross-validation
    if show_cv:
        st.subheader("Validación cruzada (5-fold)")
        try:
            if train_mode == "4 features (mejor desempeño)":
                X_for_cv = X.values
            else:
                X_for_cv = X2_all

            pipe_cv = make_pipeline(spec.estimator, use_scaler=use_scaler)
            scores = cross_val_score(pipe_cv, X_for_cv, y.values, cv=5, scoring="accuracy")
            st.write(f"Accuracy CV promedio: **{scores.mean():.3f}**  |  Desv: **{scores.std():.3f}**")
            st.write("Scores:", np.round(scores, 3))
        except Exception as e:
            st.warning(f"No se pudo calcular CV: {e}")

with tab3:
    st.subheader("Curvas (ROC multiclase)")

    if not show_roc:
        st.info("Activa 'Mostrar ROC' en la barra lateral para ver las curvas.")
    else:
        if y_score is None:
            st.warning(
                "Este modelo/configuración no pudo generar probabilidades (predict_proba) para ROC. "
                "Tip: en SVM activa probability=True."
            )
        else:
            try:
                fig_roc = plot_multiclass_roc(y_true_np, y_score, class_names)
                st.pyplot(fig_roc, use_container_width=True)
            except Exception as e:
                st.warning(f"No se pudo graficar ROC: {e}")

with tab4:
    st.subheader("Frontera de decisión (2D)")

    st.caption(
        "La frontera de decisión es 2D. Si entrenas con 4 features, aquí se entrena un modelo 2D separado solo para visualizar."
    )

    # Gráfico de frontera
    try:
        title = f"Frontera de decisión - {spec.name} ({rep_title})"
        fig_db = plot_decision_boundary_2d(
            X2_train=X2_train,
            y_train=y_train.to_numpy(),
            X2_test=X2_test,
            y_test=y_test.to_numpy(),
            model_2d=pipeline_2d,
            class_names=class_names,
            title=title,
            mesh_step=0.03,
        )
        st.pyplot(fig_db, use_container_width=True)
    except Exception as e:
        st.warning(f"No se pudo graficar la frontera de decisión: {e}")

st.sidebar.caption("▶ Ejecuta con: streamlit run main_app.py")
