# ===----------------------------------------------------------------------=== #
# Endia 2024
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from memory import ArcPointer
from algorithm import vectorize, parallelize

# from utils import Variant
from time import perf_counter
from random import seed, random_ui64
import math
from python import Python, PythonObject
from math import isnan, isinf


fn float_to_str(in_num: SIMD[dtype, 1]) -> String:
    if isinf(in_num):
        return "inf"

    if isnan(in_num):
        return "nan"

    # if (
    #     dtype == DType.uint32
    #     or dtype == DType.uint64
    #     or dtype == DType.int32
    #     or dtype == DType.int64
    # ):
    #     return str(int(in_num))

    # Determine the sign of the number
    var sign: String = ""
    var num = in_num
    if num < 0:
        sign = "-"
        num = -num

    var str_num = str(num)
    if num >= 100 and num < 999:
        if num % int(num) == 0:
            return sign + str_num[:4]
        return sign + str_num[: min(len(str_num), 8)]
    elif num >= 10 and num < 100:
        if num % int(num) == 0:
            return sign + str_num[:3]
        return sign + str_num[: min(len(str_num), 7)]
    elif num >= 1 and num < 10:
        if num % int(num) == 0:
            return sign + str_num[:2]
        return sign + str_num[: min(len(str_num), 6)]
    elif num >= 0.1 and num < 1:
        return sign + str_num[: min(len(str_num), 6)]
    elif num >= 0.01 and num < 0.1:
        return sign + str_num[: min(len(str_num), 6)]
    elif num >= 0.001 and num < 0.01:
        return sign + str_num[: min(len(str_num), 6)]
    else:
        return format_scientific(in_num)


fn format_scientific[dtype: DType](in_num: SIMD[dtype, 1]) -> String:
    if in_num == 0:
        return "0."

    # Determine the sign of the number
    var sign: String = ""
    var num = in_num
    if num < 0:
        sign = "-"
        num = -num
    else:
        sign = ""

    # Find the exponent
    var exponent: Int = 0
    while num >= 10 or (num != 0 and num < 1):
        if num >= 10:
            num /= 10
            exponent += 1
        else:
            num *= 10
            exponent -= 1

    var num_int = int(num * 10000)
    var result: String = ""
    result += sign
    result += str(num_int // 10000)
    result += "."

    # Append the decimal digits to the result string
    var decimal_digits = str(num_int % 10000)
    if len(decimal_digits) < 4:
        var padding = str("0") * (4 - len(decimal_digits))
        decimal_digits = padding + decimal_digits
    result += decimal_digits[:3]
    result += "e"
    result += str(exponent)

    return result


fn extract_array(s: String) raises -> Array:
    """
    Extracts an array from a string. If is_complex is True, expression such as 1+2i or 1+2j are allowed.\n
    If is_complex is False, only the real parts are read in.
    """
    var shape = List[Int]()
    var data = List[SIMD[dtype, 1]]()
    var iterator = -1

    fn is_digit(char: String) -> Bool:
        var res = False
        if (
            char == "0"
            or char == "1"
            or char == "2"
            or char == "3"
            or char == "4"
            or char == "5"
            or char == "6"
            or char == "7"
            or char == "8"
            or char == "9"
        ):
            res = True
        return res

    # infer shape of array by counting opening and closing brackets
    for i in range(len(s)):
        var char = s[i]
        if char == "[":
            iterator += 1
            if shape.size <= iterator:
                shape.append(0)
        elif char == "]":
            if shape[iterator - 1] == 0 and iterator > 0:
                shape[iterator] += 1
            iterator -= 1

    # read in dim of last axis
    var i = 0
    for i in range(len(s)):
        var char = s[i]
        if is_digit(char):
            var counter = 1
            for j in range(i, len(s)):
                var digit = s[j]
                if digit == "]":
                    break
                elif digit == ",":
                    counter += 1
            shape.append(counter)
            break

    # var is_complex = False

    # read in actual data as dtype
    i = 0
    while i < len(s):
        var char = s[i]
        if is_digit(char):
            # var counter = 1
            var digit: String = ""
            var start = i
            for j in range(start, len(s)):
                var c = s[j]
                if c == "]" or c == "[" or c == ",":
                    break
                else:
                    digit += c if c != " " else ""
                i += 1

            # compute number in front of comma:
            var real: String = ""
            var imag: String = ""
            var is_imag = False
            for j in range(len(digit)):
                if digit[j] == "+" or digit[j] == "i" or digit[j] == "j":
                    is_imag = True
                    continue
                if is_imag:
                    imag += digit[j]
                else:
                    real += digit[j]

            var order = 0
            for j in range(len(real)):
                if real[j] == ".":
                    break
                order += 1

            var number: SIMD[dtype, 1] = 0
            for j in range(len(real)):
                if real[j] == ".":
                    continue
                number += SIMD[dtype, 1](atol(real[j])) * SIMD[dtype, 1](
                    10
                ) ** (order - 1)

                order -= 1
            data.append(number)

        i += 1

    var new_shape = List[Int]()
    for i in range(1, shape.size):
        new_shape.append(shape[i])

    var res = Array(new_shape)
    for i in range(res.size()):
        res.store(i, data[i])
    return res


fn build_out_string(
    arg: Array, inout out: String, inout idx: Int, dim: Int, indent: String
):
    """
    Internal recursive function to build the out string for the __call__ function.
    """
    var skip_threshold = 10

    out += "["
    var row_size = arg.node[].shape[].shape[dim]

    for i in range(row_size):
        if (
            row_size > skip_threshold
            and i >= skip_threshold // 2
            and i < row_size - skip_threshold // 2
        ):
            if i == skip_threshold // 2:
                out += (
                    "...\n" + indent if dim
                    < arg.node[].shape[].ndim - 1 else "..., "
                )
            idx += (
                arg.node[].shape[].shape[dim + 1] if dim
                < arg.node[].shape[].ndim - 1 else 1
            )
            continue
        if dim < arg.node[].shape[].ndim - 1:
            build_out_string(arg, out, idx, dim + 1, indent + " ")
            if i < row_size - 1:
                out += ",\n" + indent
        else:
            out += float_to_str(arg.load(idx))
            if arg.is_complex():
                out += " + " + float_to_str(arg.load_imag(idx)) + "j"
            idx += 1
            if i < row_size - 1:
                out += ", "
    out += "]"
