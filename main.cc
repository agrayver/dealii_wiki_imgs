#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

using namespace dealii;

template<int dim>
class TestFunction: public Function<dim>
{
public:
  double value(const Point<dim> &p, const unsigned int component = 0) const
  {
    double v = 0;
    for(int d = 0; d < dim; ++d)
    {
      v += cos(2 * numbers::PI * p[d]);
    }

    return v;
  }
};

template<int dim>
void shell_grid(unsigned fe_order,
                unsigned mapping_order,
                unsigned output_mesh_order)
{
  FE_Q<dim> fe(fe_order);
  MappingQGeneric<dim> mapping(mapping_order);

  Triangulation<dim> triangulation;
  GridGenerator::hyper_shell(triangulation, Point<dim>(), 0.5, 1., 0, true);

  for(unsigned n = 0; n < 2; ++n)
  {
    for(auto cell: triangulation.active_cell_iterators())
    {
      for(unsigned face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
        if(cell->face(face)->at_boundary() &&
           cell->face(face)->boundary_id() == 1)
          cell->set_refine_flag();
    }
    triangulation.execute_coarsening_and_refinement();
  }

  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  TestFunction<dim> function;
  Vector<double> vec(dof_handler.n_dofs());
  VectorTools::interpolate(mapping, dof_handler, function, vec);

  DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = true;

  std::stringstream ss;
  ss << "shell_dim=" << dim
     << "_p=" << fe_order
     << "_mapping=" << mapping_order
     << "_n=" << output_mesh_order
     << ".vtu";

  std::ofstream out(ss.str());
  DataOut<dim> data_out;
  data_out.set_flags(flags);
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(vec, "vec");
  data_out.build_patches(mapping, output_mesh_order, DataOut<dim>::curved_inner_cells);
  data_out.write_vtu(out);
}

int main()
{
  const unsigned fe_order = 4;
  const unsigned mapping_order = 4;
  const unsigned output_mesh_order = 4;

  shell_grid<2>(fe_order, mapping_order, output_mesh_order);
  shell_grid<3>(fe_order, mapping_order, output_mesh_order);
}
