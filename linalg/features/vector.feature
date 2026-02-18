Feature: Vector math


Scenario Outline: Test whether two vectors are equal
  Given vector V1 is <v1>
  And vector V2 is <v2>
  When I compare using equality operator
  Then result should be <result_value>

Examples:
  | v1      | v2        | result_value |
  | [1,2,3] | [1,2,3]   | True         |
  | [1,2,3] | [1,3,2]   | False        |
  | [1,2,3] | [1,2,3,4] | False        |
  | [1,2,3] | [1.0,2,3] | True         |


Scenario Outline: Test Precision descriptor class
  Given I create a vector with values <values>
  When I set precision to <new_precision>
  Then the precision should be <expected_precision>

Examples:
  | values  | new_precision | expected_precision |
  | [1,2,3] | 5             | 5                  |
  | [1,2,3] | 0             | 0                  |
  | [1,2,3] | 9             | 9                  |


Scenario Outline: Test Precision descriptor validation
  Given I create a vector with values <values>
  When I try to set precision to <invalid_precision>
  Then I should get a ValueError

Examples:
  | values  | invalid_precision |
  | [1,2,3] | -1                |
  | [1,2,3] | 10                |
  | [1,2,3] | 15                |

  Scenario Outline: Test _PrecisionContext with "with" statement
  Given I create a vector with values <values>
  And I set initial precision to <initial_precision>
  When I use precision context manager with precision <context_precision>
  Then inside context precision should be <context_precision>
  And after context precision should be <initial_precision>

Examples:
  | values  | initial_precision | context_precision |
  | [1,2,3] | 2                 | 5                 |
  | [4,5,6] | 0                 | 9                 |
  | [7,8,9] | 3                 | 1                 |


Scenario: Test context manager returns vector object
  Given I create a vector with values [1,2,3]
  When I use precision context manager with precision 5
  Then the context manager should return the same vector object

