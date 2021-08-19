import numpy as np

def bisection(f, a, b, e):
    """
    f is the function
    a is one bound on the interval
    b is the other bound on the interval
    e is the maximum acceptable error
    """
    if(f(b) < f(a)):
        d = a
        a = b
        b = d
    n = int(np.log2(np.abs(b-a)) - np.log2(e) + 1)
    for i in range(0, n):
        c = (a+b)/2
        if(f(c) == 0):
            return c
        elif(f(c) > 0):
            b = c
        else:
            a = c
    return c

def false_position(f, a, b, n):
    """
    f is the function
    a is one bound on the interval
    b is the other bound on the interval
    n is the maximum number of iterations
    """
    iter = 0
    if(f(b) < f(a)):
        d = a
        a = b
        b = d
    c = (a + b)/2
    for i in range(0, n):
        iter += 1
        c = (a*f(b) - b*f(a))/(f(b) - f(a))
        if(f(c) == 0):
            print("Number of iterations = ", iter)
            return c
        elif(f(c) > 0):
            b = c
        else:
            a = c

def newton(f, df, xn, n, e, min_dfx):
    """
    f is the function whose roots are to be found.
    df is the derivative of f.
    xn is the first approximation.
    n is the maximum number of iterations.
    e is the value such that if 2 consecutive xn values are less than e apart, then terminate and return xn.
    min_dfx is the minimum acceptable absolute value of the derivative. (if |dfx| < min_dfx, terminate, return xn, tell user
    that the derivative is too close to 0, don't trust the result)
    """
    fx = f(xn)
    dfx = df(xn)
    for i in range(0, n):
        if(np.abs(dfx) < min_dfx):
            print("Derivative is too close to 0.")
            return xn
        d = fx/dfx
        xn = xn - d
        if(np.abs(d) < e):
            return xn
        fx = f(xn)
        dfx = df(xn)
        # The following line can be deleted if output for each iteration is not wanted
        # print("Iteration ", i+1, ": xn = ", xn, ", f(xn) = ", fx)
    return xn

def refined_newton(f, df, xn, n, e, min_dfx):
    """
    f is the function whose roots are to be found.
    df is the derivative of f.
    xn is the first approximation.
    n is the maximum number of iterations.
    e is the value such that if 2 consecutive xn values are less than e apart, then terminate and return xn.
    min_dfx is the minimum acceptable absolute value of the derivative. (if |dfx| < min_dfx, terminate, return xn, tell user
    that the derivative is too close to 0, don't trust the result)
    """
    fx = f(xn)
    fn = fx
    dfx = df(xn)
    for i in range(0, n):
        if(np.abs(dfx) < min_dfx):
            print("Derivative is too close to 0.")
            return xn
        d = fx/dfx
        fn = f(xn - d)
        if(np.abs(fn) > np.abs(fx)):
            xn = xn - d/2
            fn = f(xn)
        else:
            xn = xn - d
        if(np.abs(d) < e):
            return xn
        fx = fn
        dfx = df(xn)
        # The following line can be deleted if output for each iteration is not wanted
        # print("Iteration ", i+1, ": xn = ", xn, ", f(xn) = ", fx)
    return xn

def forward_elimination(A, b):
    """
    A is an (n x n) matrix.
    b is an vector of length n.
    This function takes in an (n x n) matrix A, and a vector b of length n, and carries out forward elimination to yield an
    upper triangular matrix. It changes b in an analogous way so that the equation Ax = b has the same solution before and after
    the algorithim is done. This function is meant to prepare a matrix and vector for back sub when carrying out naive gaussian 
    elimination. The function alters A and b and returns nothing.
    """
    n = np.shape(A)[0]
    for k in range(n-1):
        for i in range(k+1, n):
            m = A[i, k]/A[k, k]
            for j in range(k, n):
                A[i, j] = A[i, j] - m*A[k, j]
            b[i] = b[i] - m*b[k]

def back_sub(A, b):
    """
    A is an upper triangular (n x n) matrix.
    b is an vector of length n.
    This function carries out backwards substitution on an upper triangular matrix. It returns the vector x that is the 
    solution to the equation Ax = b. This function is meant to be called after forward elimination.
    """
    n = np.shape(A)[0]
    x = np.zeros(n)
    sum = 0
    for i in range(n-1, -1, -1):
        sum = b[i]
        for j in range(i+1, n):
            sum -= x[j]*A[i, j]
        x[i] = sum/A[i, i]
    return x

def naive_gauss(A, b):
    """
    A is an (n x n) matrix.
    b is a vector of length n.
    This function solves the equation Ax = b using naive gaussian elimination. It returns the vector x.
    """
    forward_elimination(A, b)
    return back_sub(A, b)

def gauss(A):
    """
    A is an (n x n) matrix, this function changes the A in memory.
    This function return the index array.
    This function carries out gaussian elimination with scaled partial pivoting. It takes a
    square matrix as an argument and transforms it into an upper triangular matrix ready for back 
    substitution. It does not take a constant vector b (as in Ax = b). Instead it stores the operations
    that need to be done to any given b. This is more efficient because it eliminates the need to carry
    out the entire forward elimination for every new b. The operations are stored in A in the entries
    below the diagonal.
    """
    # n is number of rows, l is index array, s is max row values (absolute values)
    n = np.shape(A)[0]
    l = np.arange(n)
    s = np.zeros(n)

    # This loop finds the max absolute value of each row and stores them to s.
    for i in range(n):
        s[i] = np.abs(A[i, 0])
        for j in range(1, n):
            s[i] = max(s[i], np.abs(A[i, j]))
    
    # main loop that does forward elimination
    for k in range(n-1):
        # Find pivot row and adjust the index array
        maxr = 0
        pivot = k
        for i in range(k, n):
            r = np.abs(A[l[i], k]/s[l[i]])
            if(r > maxr):
                maxr = r
                pivot = i
        temp = l[k]
        l[k] = l[pivot]
        l[pivot] = temp
        # Use pivot row to eliminate entries in a column.
        for i in range(k+1, n):
            # Compute multiplier for eliminating 1st entry of l[i] row.
            m = A[l[i], k]/A[l[k], k]
            # Store this multiplier in entry that is being eliminated.
            A[l[i], k] = m
            # Carry out row operation
            for j in range(k+1, n):
                A[l[i], j] -= m*A[l[k], j]
    return l

def solve(A, l, b):
    """
    A is the matrix that has already been processed by gauss(A).
    l is the index vector that gauss(A) returns.
    b is the constant vector in the equation Ax = b.
    This function returns the solution vector x.
    This function solves Ax = b and is meant to be used after gauss(A).
    This function edits b in memory, but it does not change A or l.
    """
    n = np.shape(A)[0]
    x = np.zeros(n)

    # Process the b vector with the stored values below the diagonal of A. This will change b to 
    # what it would be, had it been directly edited by the forward elimination phase.
    for k in range(n-1):
        for i in range(k+1, n):
            b[l[i]] -= A[l[i], k]*b[l[k]]

    # Start creating x
    for i in range(n-1, -1, -1):
        x[i] = b[l[i]]
        for j in range(i+1, n):
            x[i] -= A[l[i], j]*x[j]
        x[i] /= A[l[i], i]

    return x

def inverse(A):
    """
    This function takes in a square matrix and outputs its inverse. It runs gauss() once and solve() n times. 
    It solves the equation Ax = b where x is the jth column of A inverse and b is the jth column of the identity 
    matrix, for each column of A inverse. This function alters A in memory.
    """
    n = np.shape(A)[0]
    # Run gauss once in order to factor A.
    l = gauss(A)
    # This will be the inverse matrix.
    A_inverse = np.zeros((n, n))
    # Generate the jth column of A_inverse
    for j in range(n):
        # b is the jth column of the identity matrix
        b = np.zeros(n)
        b[j] = 1
        # x is the jth column of A_inverse
        x = solve(A, l, b)
        # write x out to the jth column of A_inverse
        for i in range(n):
            A_inverse[i, j] = x[i]

    return A_inverse

def tri(a, d, c, b):
    """
    a is the subdiagonal.
    d is the diagonal.
    c is the superdiagonal.
    b is the constant vector, as in Ax = b.
    returns the solution vector.
    This solves tribanded linear systems using gaussian elimination optimized for the banded structure.
    """
    n = np.shape(d)[0]
    x = np.zeros(n)
    for i in range(n-1):
        m = a[i]/d[i]
        d[i+1] -= m*c[i]
        b[i+1] -= m*b[i]

    x[n-1] = b[n-1]/d[n-1]
    for i in range(n-2, -1, -1):
        x[i] = (b[i] - c[i]*x[i+1])/d[i]

    return x

def multi_b_forward(a, d, c):
    """
    a is the subdiagonal.
    d is the diagonal.
    c is the superdiagonal.
    return a vector consisting of the multipliers needed to process a right hand side.
    Carries out the forward part of gaussian elimination for a tribanded linear 
    system of equations. This is to be used first in conjunction with multi_b_solve.
    """
    n = np.shape(d)[0]
    factors = np.zeros(n)
    for i in range(n-1):
        m = a[i]/d[i]
        d[i+1] -= m*c[i]
        factors[i] = m

    return factors

def multi_b_solve(a, d, c, b, factors):
    """
    This takes in vectors that have already been processed by multi_b_forward. This function 
    processes b, and then does backwards substituion to solve for x. This is more efficient to 
    use when you have multiple right hand sides to the same tribanded matrix, because you only 
    have to run multi_b_forward once and then you can run this function over and over with new right 
    hand sides.
    """
    # Process b
    n = np.shape(d)[0]
    x = np.zeros(n)
    for i in range(n-1):
        b[i+1] -= factors[i]*b[i]
    # back sub
    x[n-1] = b[n-1]/d[n-1]
    for i in range(n-2, -1, -1):
        x[i] = (b[i] - c[i]*x[i+1])/d[i]

    return x

def tri_pivot(a, d, c, b):
    """
    This function is the same as tri except it does scaled partial pivoting when possible.
    This function will only change the pivot if it does not mess up the tribanded structure.
    """
    n = np.shape(d)[0]
    x = np.zeros(n)
    s = np.zeros(n)
    # We need to make the scale vector
    s[0] = max(np.abs(d[0]), np.abs(c[0]))
    # This loop finds the max absolute value of each row and stores them to s.
    for i in range(1, n-1):
        s[i] = np.abs(a[i-1])
        s[i] = max(s[i], np.abs(d[i]))
        s[i] = max(s[i], np.abs(c[i]))
    s[n-1] = max(np.abs(a[n-2]), np.abs(d[n-1]))

    for i in range(n-1):
        # Find pivot row and swap rows if necessary
        if(a[i]/s[i+1] > d[i]/s[i]):
            if(((i < n-2) and (c[i+1] == 0)) or (i == n-2)):
                # swap rows, this is tedious because its not a matrix so we can't rely on index vector
                temp = a[i]
                a[i] = d[i]
                d[i] = temp
                temp = d[i+1]
                d[i+1] = c[i]
                c[i] = temp
                temp = b[i+1]
                b[i+1] = b[i]
                b[i] = temp
        
        m = a[i]/d[i]
        d[i+1] -= m*c[i]
        b[i+1] -= m*b[i]

    x[n-1] = b[n-1]/d[n-1]
    for i in range(n-2, -1, -1):
        x[i] = (b[i] - c[i]*x[i+1])/d[i]

    return x

def Bi_diagonal(n, a, d, b):
    """
    n is number of rows.
    a is off diagonal elements.
    d is diagonal.
    b is constant vector, as in Ax = b.
    This solves a very particular type of linear system where the number of equations is odd and 
    there is one entry below the diagonal for the top half and one entry above the diagonal for the 
    bottom half, meeting in the middle so the middle row has 3 entries. It is 0 everywhere else. The 
    a input contains everything off the diagonal in top to bottom left to right order.
    """
    # n is odd
    m = n//2
    x = np.zeros(n)
    # top half
    x[0] = b[0]/d[0]
    for i in range(1, m):
        x[i] = (b[i] - a[i-1]*x[i-1])/d[i]
    # bottom half
    x[n-1] = b[n-1]/d[n-1]
    for i in range(n-2, m, -1):
        x[i] = (b[i] - a[i]*x[i+1])/d[i]
    # middle entry
    x[m] = (b[m] - a[m-1]*x[m-1] - a[m]*x[m+1])/d[m]

    return x

def Backward_tri(a, d, c, b):
    # Backward_tri is just tri with the column order reversed and c swapped with a.
    n = np.shape(d)[0]
    x = tri(c, d, a, b)
    # reverse x
    r = np.zeros(n)
    for i in range(n):
        r[i] = x[n-1-i]
    return r

def jacobi(A, b, x0, tolerance, iter):
    """
    A is the matrix in Ax = b.
    b is the constant vector.
    x0 is the initial guess.
    tolerance is the maximum amount of change between iterations that is acceptable to stop at.
    iter is the maximum number of iterations.
    This function carries out Jacobi's method on a linear system.
    """
    n = np.shape(A)[0]
    x = x0.copy()
    for k in range(iter):
        x0 = x.copy()
        for i in range(n):
            sum = b[i]
            for j in range(n):
                if(j != i):
                    sum -= A[i, j]*x0[j]
            x[i] = sum/A[i, i]
        if(np.linalg.norm(x - x0) < tolerance):
            print(k, " iterations for Jacobi")
            return x
    return False

def gauss_seidel(A, b, x0, tolerance, iter):
    """
    A is the matrix in Ax = b.
    b is the constant vector.
    x0 is the initial guess.
    tolerance is the maximum amount of change between iterations that is acceptable to stop at.
    iter is the maximum number of iterations.
    This function carries out the Gauss-Seidel algorithm method on a linear system.
    """
    n = np.shape(A)[0]
    x = x0.copy()
    for k in range(iter):
        x0 = x.copy()
        for i in range(n):
            sum = b[i]
            for j in range(n):
                if(j != i):
                    sum -= A[i, j]*x[j]
            x[i] = sum/A[i, i]
        if(np.linalg.norm(x - x0) < tolerance):
            print(k, " iterations for Gauss-Seidel")
            return x
    return False

def SOR(A, b, x0, tolerance, iter, w):
    """
    A is the matrix in Ax = b.
    b is the constant vector.
    x0 is the initial guess.
    tolerance is the maximum amount of change between iterations that is acceptable to stop at.
    iter is the maximum number of iterations.
    x is the relaxation factor.
    This function carries out the SOR method on a linear system.
    """
    n = np.shape(A)[0]
    x = x0.copy()
    for k in range(iter):
        x0 = x.copy()
        for i in range(n):
            sum = b[i]
            for j in range(n):
                if(j != i):
                    sum -= A[i, j]*x[j]
            x[i] = w*sum/A[i, i] + (1-w)*x0[i]
        if(np.linalg.norm(x - x0) < tolerance):
            print(k, " iterations for SOR")
            return x
    return False

def eulers_method(f, a, b, init, steps):
    """
    f is the derivative of x(t)
    a is the start of the time interval
    b is the end of the time interval
    init is the value of x at t = a
    steps is the number of points to subdivide the interval into
    """
    h = (b - a)/steps
    values = np.zeros(steps + 1)
    values[0] = init
    t = a
    for i in range(1, steps + 1):
        values[i] = values[i - 1] + h*f(values[i - 1], t)
        t = a + i*h
    return values

def taylor_method(order, f, a, b, init, steps):
    """
    order is the order of the taylor method to use.
    f is a function that returns the derivatives of x(t) according to the 1st argument.
    f must be written specific to each IVP in the format where f(n, x, t) is the nth derivative
    of x with respect to t.
    a is the start of the time interval.
    b is the end of the time interval.
    steps is the number of points to subdivide the interval into.
    """
    h = (b - a)/steps
    values = np.zeros(steps + 1)
    values[0] = init
    t = a
    for i in range(1, steps + 1):
        values[i] = values[i - 1]
        for n in range(1, order + 1):
            values[i] += (1/np.math.factorial(n))*(h**n)*f(n, values[i - 1], t)
        t = a + i*h
    return values

def runge_kutta_4(f, a, b, alpha, steps):
    """
    f is the formula for x'(t)
    a is the start time
    b is the end time
    alpha is the initial value (aka x(a) = alpha)
    steps is the number timesteps to take to get from a to b in the interval [a, b]
    This function performs the runge-kutta 4th order method to solve the IVP
    x'(t) = f(t), x(a) = alpha, for x(t) where t is in [a, b].
    """
    values = np.zeros(steps + 1)
    values[0] = alpha
    h = (b - a)/steps
    t = a
    for i in range(steps):
        k1 = h*f(values[i], t)
        k2 = h*f(values[i] + .5*k1, t + .5*h)
        k3 = h*f(values[i] + .5*k2, t + .5*h)
        k4 = h*f(values[i] + k3, t + h)
        values[i + 1] = values[i] + (k1 + 2*k2 + 2*k3 + k4)/6
        t = a + (i + 1)*h
    return values



