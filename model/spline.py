from model.utils import *
from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)

'We implement the spline class to compute the derivative or second derivative of a piecewise function'
'The cubic spline choice might not be the best choice'

class Spline():
    def __init__(self, spline_type='spherical'):
        assert spline_type in ['cubic', 'spherical', 'spherical_cubic', 'spherical_linear']
        self.spline_type = spline_type
        self.spline = None 

    class spherical_spline():
        def __init__(self, t, x) -> None:
            assert t.shape[0] == x.shape[0]
            self.fit_points = x
            self.fit_t = t
            cos_thetas = torch.sum(x[1:]* x[:-1], dim=-1)/(torch.norm(x[1:], dim=-1) * torch.norm(x[:-1], dim=-1))
            cos_thetas = torch.clip(cos_thetas, -1, 1)
            self.thetas = torch.arccos(cos_thetas)
            self.max_t = torch.max(self.fit_t)
            self.min_t = torch.min(self.fit_t)

        def slerp(self, x0, x1, theta, a):
            return (torch.sin((1-a)*theta) * x0 + torch.sin(a*theta) * x1) / torch.sin(theta)
        
        def slerp_d(self, x0, x1, theta, a, interval):
            # derivative of slerp
            v = (-torch.cos((1-a)*theta)*x0*theta + torch.cos(a*theta)*x1*theta) / torch.sin(theta) 
            return v * (1/interval)  # for piecewise normalisation

        def slerp_dd(self, x0, x1, theta, a, interval):
            # second derivative of slerp
            a = (-torch.sin((1-a)*theta)*x0*theta**2 - torch.sin(a*theta)*x1*theta**2) / torch.sin(theta)
            return a * (1/interval)**2  # for piecewise normalisation

        def get_interval(self, t):
            # return the smallest fitted t greater than given, and largest fitted t smaller than given
            if torch.sum(t>self.max_t)>0 or torch.sum(t<self.min_t)>0:
                raise ValueError('t is out of range of the fitted spline')
            t_expanded = t.unsqueeze(1)
            fit_t_expanded = self.fit_t.unsqueeze(0)
            greater_elements = fit_t_expanded >= t_expanded
            lower_elements = fit_t_expanded <= t_expanded
            filtered_greater = torch.where(greater_elements, fit_t_expanded, float('inf'))
            filtered_lower = torch.where(lower_elements, fit_t_expanded, float('-inf'))
            t_greater = filtered_greater.min(dim=1)[0]
            t_lower = filtered_lower.max(dim=1)[0]
            idx_greater = filtered_greater.argmin(dim=1)
            idx_lower = filtered_lower.argmax(dim=1)
            t_interval = torch.stack([t_lower, t_greater], dim=1)
            t_indexes = torch.stack([idx_lower, idx_greater], dim=1)
            return t_interval, t_indexes

        def evaluate(self, t):
            output_x = []
            t_interval, t_indexes = self.get_interval(t)
            for i in range(t_interval.shape[0]):
                t_idx = t_indexes[i]
                if t_idx[0] == t_idx[1]:
                    output_x.append(self.fit_points[t_idx[0]])
                else:
                    t0, t1 = t_interval[i]
                    x0, x1 = self.fit_points[t_idx]
                    a = (t[i] - t0) / (t1 - t0)
                    output_x.append(self.slerp(x0, x1, self.thetas[t_idx[0]], a))
            return torch.stack(output_x)
        
        def derivative_slerp(self, t, derivative_function):
            output_v = []
            t_interval, t_indexes = self.get_interval(t)
            for i in range(t_interval.shape[0]):
                t_idx = t_indexes[i]
                t0, t1 = t_interval[i]
                x0, x1 = self.fit_points[t_idx]
                idx0, idx1 = t_idx
                if idx0 == idx1:
                    if t0 == self.max_t:
                        v = derivative_function(self.fit_points[idx0-1], x0, self.thetas[idx0-1], 1, (t0-self.fit_t[idx0-1]))
                        output_v.append(v)
                    elif t0 == self.min_t:
                        v = derivative_function(x1, self.fit_points[idx1+1], self.thetas[idx0], 0, (self.fit_t[idx1+1]-t1))
                        output_v.append(v)
                    else:
                        v = 0.5*derivative_function(self.fit_points[idx0-1], x0, self.thetas[idx0-1], 1, (t0-self.fit_t[idx0-1])) \
                            + 0.5*derivative_function(x1, self.fit_points[idx1+1], self.thetas[idx0], 0, (self.fit_t[idx1+1]-t1))
                        output_v.append(v)
                else:
                    a = (t[i] - t0) / (t1 - t0)
                    output_v.append(derivative_function(x0, x1, self.thetas[idx0], a, (t1-t0)))
            return torch.stack(output_v) 

        def derivative(self, t, order=0):
            if order == 0:
                return self.evaluate(t)
            if order == 1:
                return self.derivative_slerp(t, self.slerp_d)
            if order == 2:
                return self.derivative_slerp(t, self.slerp_dd)

    class spherical_cubic_spline(spherical_spline):
        # It is a piecewise slerp function, but for the gamma_ddot, we use cubic spline to estimate
        def __init__(self, t, x) -> None:
            super().__init__(t,x)
            self.cubic_coeffs = natural_cubic_spline_coeffs(t, x)
            self.cubic_euclidean = NaturalCubicSpline(self.cubic_coeffs)
        
        def dd_slerp_cubic(self, t):
            t_interval, t_indexes = self.get_interval(t)
            t_in_fit_mask = []
            for i in range(t_interval.shape[0]):
                if t_indexes[i][0] == t_indexes[i][1]:
                    t_in_fit_mask.append(True)
                else:
                    t_in_fit_mask.append(False)
            t_in_fit_mask = torch.tensor(t_in_fit_mask)
            cubic_dd = self.cubic_euclidean.derivative(t, order=2)
            return cubic_dd

        def derivative(self, t, order=0):
            if order == 0:
                return self.evaluate(t)
            if order == 1:
                return self.derivative_slerp(t, self.slerp_d)
            if order == 2:
                # In the bvp implementation, its only the projection of such the acceleration on the tangent direction
                return self.dd_slerp_cubic(t)
    
    class finite_difference_spline(spherical_spline):
        def __init__(self, t, x) -> None:
            super().__init__(t, x)
            

    def fit_spline(self, t, x):
        if self.spline_type == 'cubic':
            coeffs = natural_cubic_spline_coeffs(t, x)
            spline = NaturalCubicSpline(coeffs)
            self.spline = spline
        elif self.spline_type == 'spherical':
            self.spline = self.spherical_spline(t, x)
        else: 
            self.spline = self.spherical_cubic_spline(t, x)

    def __call__(self, t, order=0):
        if order==0:
            return self.spline.evaluate(t)
        else:
            return self.spline.derivative(t, order)


