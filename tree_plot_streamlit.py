# Purpose: Create a tree plot using streamlit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import dtreeviz
import base64

@st.cache_data
def load_data():
    df = pd.read_csv('./data/Iris.csv')
    df = df.drop(['Id'], axis=1)
    return df

@st.cache_data
def display_data(df):
    fig, ax = plt.subplots()
    ax.scatter(df['SepalLengthCm'], df['SepalWidthCm'], c=df['Species'].astype('category').cat.codes, cmap='viridis')
    ax.set_xlabel('Sepal Length')
    ax.set_ylabel('Sepal Width')
    st.pyplot(fig)

def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)
    
def train_decision_tree_classifier(data, target, test_size=0.2, random_state=42):
    # Split data
    print("data", data.columns)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=random_state)

    # Fit model
    # use st setting the max_depth
    st.session_state.max_depth = st.slider("max_depth", 1, 10, 2)
    model = DecisionTreeClassifier(max_depth=st.session_state.max_depth)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # Evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    st.write("Accuracy: %.2f%%" % (accuracy * 100.0))
    return model
    

def svg_write(svg, center=True):
    """
    Disable center to left-margin align like other objects.
    """
    # Encode as base 64
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")

    # Add some CSS on top
    css_justify = "center" if center else "left"
    css = f'<p style="text-align:center; display: flex; justify-content: {css_justify};">'
    html = f'{css}<img src="data:image/svg+xml;base64,{b64}"/>'

    # Write the HTML
    st.write(html, unsafe_allow_html=True)

def create_viz_model(model, feature, label, feature_name):
    viz_model = dtreeviz.model(model,
                           X_train=feature, y_train=label,
                           feature_names=feature_name,
                           target_name='iris',
                           class_names=['setosa', 'versicolor', 'virginica'])
    return viz_model

# display important features
def display_importance(model, feature_name):
    importance = model.feature_importances_
    # use st show the feature importance, use feature name as index, list
    importance_df = pd.DataFrame(importance, index=feature_name, columns=['importance']).sort_values('importance', ascending=False)
    st.write(importance_df)
    return importance_df
    
def display_checkbox(items):
    ## default checkbox is true
   checkbox_states = {}
   for item in items:
      checkbox_states[item] = st.checkbox(item, value=True)
   selected_items = [item for item, state in checkbox_states.items() if state]
#    st.write(selected_items)    
    # selected_items = {}
    # for item, state in checkbox_states.items():
    #     if state:
    #         selected_items[item] = state
   return selected_items

def show_modle_result(model, feature, label):
    viz_model = create_viz_model(model, feature, label, feature.columns)
    vizRender = viz_model.view(scale=1)
    vizRender.save("play_tennis_decision_tree.svg")
    st.image(vizRender._repr_svg_(), use_column_width=True)
    return 

def main():
    st.write("""
    # Iris Flower Prediction App
    This app predicts the **Iris flower** type!
    """)
    # Import data
    df = load_data()
    # display_data(df)
    feature = df.drop(['Species'], axis=1)
    feature_name = df.drop(['Species'], axis=1).columns
    label = df['Species'].astype('category').cat.codes
    
    st.write("Feature Selection")
    st.session_state.select_feature = display_checkbox(feature.columns)
    
    if 'select_feature' not in st.session_state:
        st.session_state.select_feature = feature_name

    model = train_decision_tree_classifier(feature[st.session_state.select_feature], label)
    _ =  display_importance(model, st.session_state.select_feature)
    show_modle_result(model, feature[st.session_state.select_feature], label)


if __name__ == "__main__":
    main()