#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "include/signalsmith-stretch.h"

namespace py = pybind11;
using namespace pybind11::literals;

class SignalArrayWrapper
{
private:
    py::ssize_t channel_;
    py::detail::unchecked_mutable_reference<double, 2> &array_;

public:
    SignalArrayWrapper(py::ssize_t channel, py::detail::unchecked_mutable_reference<double, 2> &array) : channel_(channel), array_(array) {}

    /// Default constructor (disabled)
    SignalArrayWrapper() = delete;

    /// Copy constructor (disabled)
    SignalArrayWrapper(const SignalArrayWrapper &) = delete;

    /// Assignment operator (disabled)
    void operator=(const SignalArrayWrapper &) = delete;

    double &operator[](std::size_t i) { return array_(channel_, i); }
};

class SignalStretch
{
private:
    signalsmith::stretch::SignalsmithStretch<double> stretch_;
    double time_stretch_factor = 1.0;

public:
    SignalStretch() {}

    SignalStretch(long seed) : stretch_(seed) {}

    /// Copy constructor (disabled)
    SignalStretch(const SignalStretch &) = delete;

    /// Assignment operator (disabled)
    void operator=(const SignalStretch &) = delete;

    void configure_preset(int channels, double sample_rate, bool cheap = false)
    {
        if (cheap)
            stretch_.presetCheaper(channels, sample_rate);
        else
            stretch_.presetDefault(channels, sample_rate);
    }

    void configure_custom(int channels, int block_samples, int interval_samples)
    {
        stretch_.configure(channels, block_samples, interval_samples);
    }

    void set_transpose_factor(double multiplier, double tonality_limit = 0.0)
    {
        stretch_.setTransposeFactor(multiplier, tonality_limit);
    }

    void set_transpose_semitones(double semitones, double tonality_limit = 0.0)
    {
        stretch_.setTransposeSemitones(semitones, tonality_limit);
    }

    void set_transpose_cents(double cents, double tonality_limit = 0.0)
    {
        stretch_.setTransposeSemitones(cents / 100.0, tonality_limit);
    }

    void set_stretch_factor(double factor)
    {
        time_stretch_factor = factor;
    }

    py::array_t<double> process(py::array_t<double> input)
    {
        if (input.ndim() <= 1)
            input = input.reshape({py::ssize_t_cast(1), input.size()});

        py::ssize_t output_length = py::ssize_t_cast(input.shape(1) / time_stretch_factor);
        py::array_t<double> output({py::ssize_t_cast(1), output_length});

        auto input_access = input.mutable_unchecked<2>();
        auto output_access = output.mutable_unchecked<2>();

        SignalArrayWrapper input_buffer[2] = {{0, input_access}, {1, input_access}};
        SignalArrayWrapper output_buffer[2] = {{0, output_access}, {1, output_access}};

        stretch_.process(input_buffer, input.shape(1), output_buffer, output_length);

        return output;
    }

    void reset()
    {
        stretch_.reset();
    }
};

PYBIND11_MODULE(stretch, m)
{
    py::class_<SignalStretch>(m, "SignalStretch")
        .def(py::init<>())
        .def(py::init<long>(), "seed"_a)
        .def("configure_preset", &SignalStretch::configure_preset, "channels"_a, "sample_rate"_a, "cheap"_a = false)
        .def("configure_custom", &SignalStretch::configure_custom, "channels"_a, "block_samples"_a, "interval_samples"_a)
        .def("set_transpose_factor", &SignalStretch::set_transpose_factor, "multiplier"_a, "tonality_limit"_a = 0.0)
        .def("set_transpose_semitones", &SignalStretch::set_transpose_semitones, "semitones"_a, "tonality_limit"_a = 0.0)
        .def("set_transpose_cents", &SignalStretch::set_transpose_cents, "cents"_a, "tonality_limit"_a = 0.0)
        .def("set_stretch_factor", &SignalStretch::set_stretch_factor, "factor"_a)
        .def("process", &SignalStretch::process, "input_audio"_a)
        .def("reset", &SignalStretch::reset);
}
