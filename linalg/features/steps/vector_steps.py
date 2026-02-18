"""
This file describes, maps programatically steps that are described 
in file vector.feature using Gherkin syntax 
"""
import ast
from behave import given, when, then
from linalg.vector import Vector

# pylint: disable=not-callable, protected-access

# Test for Vector equality sign

@given("vector V1 is {values}")
def step_vector_v1(context, values):
    """
    Step for creating vector V1
    
    :param context: This object is passed automatically be behave so that 
    inside function dynamic attibutes and its values can be set
    :param values: String matched by the behave in the text in file vector.feature
    """
    context.v1 = Vector(ast.literal_eval(values), len(ast.literal_eval(values)))

@given("vector V2 is {values}")
def step_vector_v2(context, values):
    """
    Step for creating vector V2
    
    :param context: -"-
    :param values: -"-
    """
    context.v2 = Vector(ast.literal_eval(values), len(ast.literal_eval(values)))

@when("I compare using equality operator")
def step_add(context):
    """
    Docstring for step_add
    
    :param context: -"-
    """
    context.result = context.v1 == context.v2

@then("result should be {values}")
def step_result(context, values):
    """
    Docstring for step_result
    
    :param context: -"-
    :param values: -"-
    """
    expected = ast.literal_eval(values)
    assert context.result == expected

# Tests for Precision descriptor class

@given("I create a vector with values {values}")
def step_create_vector(context, values):
    """Step for creating a vector"""
    vector_values = ast.literal_eval(values)
    context.vector = Vector(vector_values, len(vector_values))

@given("I set initial precision to {precision}")
def step_set_initial_precision(context, precision):
    """Step for setting initial precision"""
    context.vector._precision = int(precision)

@when("I set precision to {new_precision}")
def step_change_precision(context, new_precision):
    """Step for changing precision to new value"""
    context.vector._precision = int(new_precision)

@then("the precision should be {expected_precision}")
def step_check_precision_set(context, expected_precision):
    """Step for checking if precision was set correctly"""
    expected = int(expected_precision)
    assert context.vector._precision == expected

# Tests for Precision descriptor class with wrong values

@when("I try to set precision to {precision}")
def step_try_set_invalid_precision(context, precision):
    """Step for trying to set invalid precision"""
    try:
        context.vector._precision = int(precision)
        context.exception = None
    except ValueError as e:
        context.exception = e

@then("I should get a ValueError")
def step_check_value_error(context):
    """Step for checking if ValueError was raised"""
    assert context.exception is not None
    assert isinstance(context.exception, ValueError)

# Test for _PrecisionContext with "with" statement

@when("I use precision context manager with precision {precision}")
def step_use_precision_context(context, precision):
    """Step for testing precision context manager"""
    precision_value = int(precision)
    context.precision_before_context = context.vector._precision

    with context.vector.precision(precision_value) as vector_in_context:
        context.precision_inside_context = vector_in_context._precision
        context.vector_in_context = vector_in_context

    context.precision_after_context = context.vector._precision

@then("inside context precision should be {expected_precision}")
def step_check_context_precision(context, expected_precision):
    """Step for checking precision inside context"""
    expected = int(expected_precision)
    assert context.precision_inside_context == expected

@then("after context precision should be {expected_precision}")
def step_check_restored_precision(context, expected_precision):
    """Step for checking precision after context"""
    expected = int(expected_precision)
    assert context.precision_after_context == expected

#Test context manager returns vector object

@then("the context manager should return the same vector object")
def step_check_context_returns_vector(context):
    """Step for checking if context manager returns the vector"""
    assert context.vector_in_context is context.vector
