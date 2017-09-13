from typing import Dict
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)


def scope_name():
    """Returns the name of current variable scope as a string"""
    return tf.get_variable_scope().name


def absolute_scope_name(relative_scope_name):
    """Appends parent scope name to `relative_scope_name`"""
    if scope_name() == "":
        return relative_scope_name
    return scope_name() + "/" + relative_scope_name


def _get_variables_from_scope(scope, collection):
    with tf.variable_scope(scope) as scope:
        pass

    variables = tf.get_collection(collection, scope=scope.name+"/")

    return set(filter(lambda x: isinstance(x, tf.Variable), variables))


def _variable_dictionary(collection, scope) -> Dict[str, tf.Variable]:
    with tf.variable_scope(scope) as scope:
        pass

    variables = _get_variables_from_scope(scope=scope, collection=collection)
    offset = len(scope.name)

    def var_name(var: tf.Variable):
        shortened = var.name[offset:].split(":")[0]
        if shortened.startswith("/"):
            return shortened[1:]
        return shortened

    return {var_name(var): var for var in filter(lambda x: isinstance(x, tf.Variable), variables)}


def _apply_to_variable_pairs(source, target, operation, log_message="%s <- %s"):
    source_vars = _variable_dictionary(tf.GraphKeys.GLOBAL_VARIABLES, scope=source)
    target_vars = _variable_dictionary(tf.GraphKeys.GLOBAL_VARIABLES, scope=target)

    operations = []
    with tf.name_scope("apply_to_pairs", values=list(source_vars.values()) + list(target_vars.values())):
        for target in target_vars:
            source_var = source_vars.get(target, None)
            target_var = target_vars[target]
            if source_var is None:
                logger.warning("Variable %s not found in source variables for target %s", target, target_var.name)
            else:
                with tf.name_scope(target):
                    operations.append(operation(source_var, target_var))
                logger.info(log_message, target_var.name, source_var.name)

    return tf.group(*operations)


def assign_from_scope(source_scope, target_scope, name=None) -> tf.Operation:
    """
    Assigns all variables in `target_scope`, that are also present in `source_scope`,
    the value of the original variables in `source_scope`. Scopes can be either
    strings of tf.VariableScope.
    If a variable is present in `target_scope` but not corresponding variable is found
    in `source_scope`, a warning is logged.
    :param source_scope: Scope from which to read the original values.
    :param target_scope: Scope in which the assignment targets reside.
    :param name: Name for the operation. Defaults to "assign_from_scope"
    :return: An operation that performs all assignments.
    """
    def assign(source: tf.Variable, target: tf.Variable):
        return target.assign(source)
    with tf.name_scope(name, "assign_from_scope"):
        return _apply_to_variable_pairs(source_scope, target_scope, assign, "Assigning %s <- %s")


def update_from_scope(source_scope, target_scope, rate, name=None) -> tf.Operation:
    def update(source: tf.Variable, target: tf.Variable):
        new_val = tf.constant(1.0 - rate) * target + rate * source
        return target.assign(new_val)

    with tf.name_scope(name, "soft_update"):
        return _apply_to_variable_pairs(source_scope, target_scope, update, "Updating %s <- %s")


def copy_variables_to_scope(source_scope, target_scope, trainable=None):
    """
    Copies all variables from `source_scope` to `target_scope`. The new variables will have the same
    shape, data type and initial content as the originals.
    :param source_scope: Scope from which the variables are gathered. Can be either a scope name or a
                        tf.VariableScope.
    :param target_scope: Scope to which the new variables are created
    :param bool trainable: Whether the new variables should be added to the list of trainable variables.
    :return Tuple[tf.VariableScope, List]: The variable scope in which the new variables
        were created, and a list of the new variables.
    """
    with tf.variable_scope(source_scope) as source_scope:
        pass

    sources = _variable_dictionary(collection=tf.GraphKeys.GLOBAL_VARIABLES, scope=source_scope.name)
    new_variables = []

    with tf.variable_scope(target_scope, reuse=False) as destination_scope: # type: tf.VariableScope
        with tf.name_scope(destination_scope.original_name_scope):
            for source_name in sources:
                newvar = tf.get_variable(name=source_name,
                                         initializer=sources[source_name].initialized_value(),
                                         trainable=trainable)
                new_variables.append(newvar)

    return destination_scope, new_variables
