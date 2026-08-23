from flask import Blueprint

# Create a blueprint for the routes
routes_bp = Blueprint('routes', __name__)

# Define your route handlers here
@routes_bp.route('/example', methods=['GET'])
def example_route():
    return {"message": "This is an example route."}

# You can add more routes as needed
