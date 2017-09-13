import tensorflow as tf


def scope_name():
    """Returns the name of current variable scope as a string"""
    return tf.get_variable_scope().name


def absolute_scope_name(relative_scope_name):
    """Appends parent scope name to `relative_scope_name`"""
    if scope_name() == "":
        return relative_scope_name
    return scope_name() + "/" + relative_scope_name


def assign_from_scope(source_scope, target_scope, name=None):
    # convert scope arguments to string
    if isinstance(source_scope, tf.VariableScope):
        source_scope = source_scope.name
    if isinstance(target_scope, tf.VariableScope):
        target_scope = target_scope.name

    source_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=source_scope)
    target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_scope)
    asgns = []
    with tf.name_scope(name, "assign_from_scope", values=source_vars+target_vars):
        for source in source_vars:
            for target in target_vars:
                source_name = source.name[len(source_scope):]
                target_name = target.name[len(target_scope):]
                if source_name == target_name:
                    asgns.append(target.assign(source))
        return tf.group(*asgns)


def update_from_scope(source_scope, target_scope, rate, name="soft_update"):
    if isinstance(source_scope, tf.VariableScope):
        source_scope = source_scope.name
    if isinstance(target_scope, tf.VariableScope):
        target_scope = target_scope.name

    source_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=source_scope)
    target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_scope)
    asgns = []
    with tf.name_scope(name):
        for source in source_vars:
            for target in target_vars:
                source_name = source.name[len(source_scope)+1:]
                target_name = target.name[len(target_scope)+1:]
                if source_name == target_name:
                    sn = source_name.split(":")[0]
                    with tf.name_scope(sn):
                        newval = (1 - rate) * target + rate * source
                        asgns.append(target.assign(newval))
        return tf.group(*asgns)


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
    if isinstance(source_scope, tf.VariableScope):
        source_scope = source_scope.name

    sources = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=source_scope)
    new_variables = []

    print(sources)

    with tf.variable_scope(target_scope, reuse=False) as destination_scope: # type: tf.VariableScope
        with tf.name_scope(destination_scope.original_name_scope):
            for var in sources: # type: tf.Variable
                source_name = var.name[len(source_scope):]
                if source_name.startswith("/"):
                    source_name = source_name[1:]
                source_name = source_name.split(":")[0]
                newvar = tf.get_variable(name=source_name,
                                         initializer=var.initialized_value(),
                                         trainable=trainable)
                new_variables.append(newvar)

    return destination_scope, new_variables
