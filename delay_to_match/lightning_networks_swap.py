import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression  # Import the classifier

class dDMTSNet(pl.LightningModule):
    """distractedDelayedMatchToSampleNetwork. Class defines RNN for solving a
    distracted DMTS task. Implemented in Pytorch Lightning to enable smooth
    running on multiple GPUs. """

    def __init__(
        self,
        rnn_type,
        input_size,
        hidden_size,
        output_size,
        dt_ann,
        alpha,
        alpha_W,
        g,
        nl,
        lr,
        include_delay,
        plot=False,
        hemisphere=None,
        mode='no-swap'
    ):
        super().__init__()
        self.stsp = False
        self.dt_ann = dt_ann
        self.lr = lr
        self.save_hyperparameters()
        self.act_reg = 0
        self.param_reg = 0
        self.accumulated_accuracies = {}
        self.accumulated_accuracies_no_distractor = {}
        self.accumulated_accuracies_distractor = {}
        self.include_delay = include_delay
        self.mark_delay_end_dict = {233:133, 194:94, 166:66, 366:266, 288:188}
        self.mark_test_end_dict = {233:166, 194:127, 366:300, 288:222, 166:100}
        self.mark_dis_on_dict = {233:66, 194:47, 366:133, 288:94, 166:33}
        self.mark_dis_off_dict = {233:83, 194:63, 366:150, 288:111, 166:50}
        self.rnn_type = rnn_type
        self.plot = plot
        

        if rnn_type == "vRNN":
            # if model is vanilla RNN
            self.rnn = vRNNLayer(input_size, hidden_size,
                                 output_size, alpha, g, nl,
                                 hemisphere, mode)
            self.fixed_syn = True

        if rnn_type == "ah":
            # if model is anti-Hebbian
            self.rnn = aHiHRNNLayer(
                input_size, hidden_size, output_size, alpha, alpha_W, nl
            )
            self.fixed_syn = False

        if rnn_type == "stsp":
            # if model is Mongillo/Masse STSP
            self.rnn = stspRNNLayer(
                input_size, hidden_size, output_size, alpha, dt_ann, g, nl, 
                hemisphere
            )
            self.fixed_syn = False
            self.stsp = True
            
        self.all_out_hidden_no_swap = []
        self.all_out_hidden_swap = []
     
    def set_mode(self, mode):
        """
        Set the active hemisphere for training.
        'left' for left hemisphere, 'right' for right hemisphere, 'both' for both hemispheres.
        """
        assert mode in ['no-swap', 'swap'], "Invalid mode"
        self.mode = mode
        self.rnn.set_mode(mode)
        print(f'Reset mode to {mode}')       

    def forward(self, x):
        print('im called watch out')
        breakpoint()
        # defines foward method using the chosen RNN type
        out_readout, out_hidden, w_hidden, _ = self.rnn(x)
        return out_readout, out_hidden, w_hidden, _

    def training_step(self, batch, batch_idx):
        # print('training.')
        # training_step defined the train loop.
        # It is independent of forward

        inp, out_des, y, test_on, dis_bool, samp_off, dis_on = batch
        out_readout, out_hidden, _, _ = self.rnn(inp, test_on, dis_on)

        # accumulate losses. if penalizing activity, then add it to the loss
        if self.act_reg != 0:
            loss = self.act_reg*out_hidden.norm(p='fro')
            loss /= out_hidden.shape[0]*out_hidden.shape[1]*out_hidden.shape[2]
        else:
            loss = 0

        for i in test_on.unique():
            inds = torch.where(test_on == i)[0]
            test_end = int(i) + int(500 / self.dt_ann)
            response_end = test_end + int(500 / self.dt_ann)
            loss += F.mse_loss(
                out_readout[inds, test_end:response_end],
                out_des[inds, test_end:response_end],
            )
        return loss

    def validation_step(self, batch, batch_idx):
        # print('validation')
        # defines validation step
        inp, out_des, y, test_on, dis_bool, _, dis_on = batch
        out_readout, _, _, _ = self.rnn(inp, test_on, dis_on)

        accs = np.zeros(out_readout.shape[0])
        # test model performance
        for i in range(out_readout.shape[0]):
            curr_max = (
                out_readout[
                    i,
                    int(test_on[i])
                    + int(500 / self.dt_ann): int(test_on[i])
                    + 2 * int(500 / self.dt_ann),
                    :-1,
                ]
                .argmax(dim=1)
                .cpu()
                .detach()
                .numpy()
            )
            accs[i] = (y[i].item() == curr_max).sum() / len(curr_max)

        self.log("val_acc", accs.mean(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        # print('testing')
        # Here we just reuse the validation_step for testing
        # return self.validation_step(batch, batch_idx)
        
        inp, out_des, y, test_on, dis_bool, samp_off, dis_on = batch
        out_readout, out_hidden, _, _ = self.rnn(inp, test_on, dis_on)
        if self.mode=='no-swap':
            self.all_out_hidden_no_swap.append(out_hidden.detach().cpu())
        elif self.mode == 'swap':
            self.all_out_hidden_swap.append(out_hidden.detach().cpu())

        if self.plot:
            unique_test_on_values = torch.unique(test_on)
            for test_on_value in unique_test_on_values:
                mask = test_on == test_on_value
                inp_sub, out_des_sub, y_sub, test_on_sub, dis_bool_sub, samp_off_sub = [tensor[mask] for tensor in batch]
                before_match_hidden = out_hidden[torch.arange(out_hidden.shape[0]).to(out_hidden.device)[mask], (samp_off_sub - 1).long(), :]

                clf_all = LogisticRegression()
                clf_no_distractor = LogisticRegression()
                clf_distractor = LogisticRegression()
                
                # Separate data based on distractor presence
                no_distractor_mask = dis_bool_sub == 0
                distractor_mask = dis_bool_sub == 1
            
                clf_all.fit(before_match_hidden.cpu().numpy(), y_sub.cpu().numpy())
                if no_distractor_mask.any():
                    clf_no_distractor.fit(before_match_hidden[no_distractor_mask].cpu().numpy(), y_sub[no_distractor_mask].cpu().numpy())
                if distractor_mask.any():
                    clf_distractor.fit(before_match_hidden[distractor_mask].cpu().numpy(), y_sub[distractor_mask].cpu().numpy())

                if self.include_delay: # start from delay start, which is samp_off_sub to afer
                    unique_value = int(torch.unique(samp_off_sub).item())

                else: # start from test_on
                    unique_value = int(test_on_sub[0].item())

                
                after_match_hidden = out_hidden[mask, unique_value:, :]
                int_test_on_value = int(test_on_value.item())  # Convert to int once to reuse   
                

                # Separate storage for accuracies
                for clf, acc_dict, label, mask in zip([clf_all, clf_no_distractor, clf_distractor], 
                                                [self.accumulated_accuracies, 
                                                self.accumulated_accuracies_no_distractor, 
                                                self.accumulated_accuracies_distractor], 
                                                ['all', 'no_distractor', 'distractor'],
                                                [slice(None), no_distractor_mask, distractor_mask]):
                    accuracies = []  # List to store accuracies at each time step after the match
                    for t in range(after_match_hidden.shape[1]):
                        current_time_step_hidden = after_match_hidden[mask, t, :]
                        y_sub_masked = y_sub[mask]
                        predictions = clf.predict(current_time_step_hidden.cpu().numpy())
                        accuracy = (predictions == y_sub_masked.cpu().numpy()).sum()/len(predictions)
                        accuracies.append(accuracy)

                    # Store accuracies
                    if int_test_on_value not in acc_dict:
                        acc_dict[int_test_on_value] = np.array(accuracies)
                    else:
                        acc_dict[int_test_on_value] = np.vstack((acc_dict[int_test_on_value], np.array(accuracies)))

        """compute accuracy"""
        accs = np.zeros(out_readout.shape[0])
        # test model performance
        for i in range(out_readout.shape[0]):
            curr_max = (
                out_readout[
                    i,
                    int(test_on[i])
                    + int(500 / self.dt_ann): int(test_on[i])
                    + 2 * int(500 / self.dt_ann),
                    :-1,
                ]
                .argmax(dim=1)
                .cpu()
                .detach()
                .numpy()
            )
            accs[i] = (y[i].item() == curr_max).sum() / len(curr_max)

        self.log("test_acc", accs.mean(), prog_bar=True)
        
            

    def configure_optimizers(self):
        # by default, we use an L2 weight decay on all parameters.
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.param_reg)

        # lr_scheduler = {'scheduler':  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1),"monitor": 'val_acc'}
        return [optimizer]  # ,[lr_scheduler]

    def on_test_epoch_end(self):
        if self.mode == 'no-swap':
            all_out_hidden_combined = torch.cat(self.all_out_hidden_no_swap, dim=0)
            torch.save(all_out_hidden_combined, 'baseline_result/out_hidden_combined_no_swap.pt')
            print('saved hidden rep of no-swap')
            
        elif self.mode == 'swap':
            all_out_hidden_combined = torch.cat(self.all_out_hidden_swap, dim=0)
            torch.save(all_out_hidden_combined, 'baseline_result/out_hidden_combined_swap.pt')
            print('saved hidden rep of swap')
            
        if self.plot:
            for test_on_value in self.accumulated_accuracies.keys():
                # Get the data for this test_on value
                acc_all = self.accumulated_accuracies[test_on_value]
                acc_no_distractor = self.accumulated_accuracies_no_distractor.get(test_on_value, None)
                acc_distractor = self.accumulated_accuracies_distractor.get(test_on_value, None)

                # Calculate mean and SEM
                Y_mean_all = acc_all.mean(axis=0)
                Y_sem_all = acc_all.std(axis=0) / np.sqrt(acc_all.shape[0])

                # Create a new figure for each test_on value
                plt.figure()
                x = np.arange(len(Y_mean_all))  # Assuming x-axis is the index of time steps post-match

                # Plot data for all cases
                plt.plot(x, Y_mean_all, label='All Data')
                plt.fill_between(x, Y_mean_all - Y_sem_all, Y_mean_all + Y_sem_all, alpha=0.5)
                
                mark_delay_end = self.mark_delay_end_dict.get(test_on_value, None)
                mark_test_end = self.mark_test_end_dict.get(test_on_value, None)
                mark_dis_on = self.mark_dis_on_dict.get(test_on_value, None)
                mark_dis_off = self.mark_dis_off_dict.get(test_on_value, None)
                
                plt.axvline(x=0, color='gray', linestyle='--', label='Delay Start')
                plt.axvline(x=mark_dis_on, color='khaki', linestyle='-', label='Distractor Start')
                plt.axvline(x=mark_dis_off, color='khaki', linestyle='--', label='Distractor End')
                if mark_delay_end is not None:
                    plt.axvline(x=mark_delay_end, color='gray', linestyle='-', label='Delay End/Test Start')
                if mark_test_end is not None:
                    plt.axvline(x=mark_test_end, color='lightsteelblue', linestyle='--', label='Test End')

                # If there's data for no distractor, plot it
                if acc_no_distractor is not None:
                    Y_mean_no_distractor = acc_no_distractor.mean(axis=0)
                    Y_sem_no_distractor = acc_no_distractor.std(axis=0) / np.sqrt(acc_no_distractor.shape[0])
                    plt.plot(x, Y_mean_no_distractor, label='Without Distractor')
                    plt.fill_between(x, Y_mean_no_distractor - Y_sem_no_distractor, Y_mean_no_distractor + Y_sem_no_distractor, alpha=0.5)

                # If there's data for distractor, plot it
                if acc_distractor is not None:
                    Y_mean_distractor = acc_distractor.mean(axis=0)
                    Y_sem_distractor = acc_distractor.std(axis=0) / np.sqrt(acc_distractor.shape[0])
                    plt.plot(x, Y_mean_distractor, label='With Distractor')
                    plt.fill_between(x, Y_mean_distractor - Y_sem_distractor, Y_mean_distractor + Y_sem_distractor, alpha=0.5)

                # Customize and save the plot
                plt.xlabel('Time steps post-delay')
                plt.ylabel('Accuracy')
                plt.title(f'Test On: {test_on_value}')
                plt.legend()
                plt.savefig(f'results/{self.rnn_type}/convergence_{test_on_value}_sampleoff_teston_diffcases_includeDelay.pdf')
                plt.show()

            # Reset accumulated accuracies for the next testing epoch
            self.accumulated_accuracies = {}
            self.accumulated_accuracies_no_distractor = {}
            self.accumulated_accuracies_distractor = {}

class vRNNLayer(pl.LightningModule):
    """Vanilla RNN layer in continuous time."""

    def __init__(self, input_size, hidden_size, output_size, alpha, g, nonlinearity,
                 hemisphere, mode):
        super(vRNNLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.alpha = alpha
        self.inv_sqrt_alpha = 1 / np.sqrt(alpha)
        self.cont_stab = False
        self.disc_stab = True
        self.g = g
        self.process_noise = 0.05

        # set nonlinearity of the vRNN
        self.nonlinearity = nonlinearity
        if nonlinearity == "tanh":
            self.phi = torch.tanh
        if nonlinearity == "relu":
            self.phi = F.relu
        if nonlinearity == "none":
            print("Nl = none")
            self.phi = torch.nn.Identity()

        # initialize the input-to-hidden weights
        self.weight_ih = nn.Parameter(
            torch.normal(0, 1 / np.sqrt(hidden_size),
                         (hidden_size, input_size))
        )

        # initialize the hidden-to-output weights
        self.weight_ho = nn.Parameter(
            torch.normal(0, 1 / np.sqrt(hidden_size),
                         (output_size, hidden_size))
        )

        # initialize the hidden-to-hidden weights
        self.W = nn.Parameter(
            torch.normal(0, self.g / np.sqrt(hidden_size),
                         (hidden_size, hidden_size))
        )

        # initialize the output bias weights
        self.bias_oh = nn.Parameter(
            torch.normal(0, 1 / np.sqrt(hidden_size), (1, output_size))
        )

        # initialize the hidden bias weights
        self.bias_hh = nn.Parameter(
            torch.normal(0, 1 / np.sqrt(hidden_size), (1, hidden_size))
        )

        # define mask for weight matrix do to structural perturbation experiments
        self.struc_p_0 = 0
        self.register_buffer(
            "struc_perturb_mask",
            torch.FloatTensor(self.hidden_size, self.hidden_size).uniform_()
            > self.struc_p_0,
        )
        
        left_mask_for_weight_ih, right_mask_for_weight_ih = torch.zeros_like(self.weight_ih), torch.zeros_like(self.weight_ih)
        left_mask_for_weight_ho, right_mask_for_weight_ho = torch.zeros_like(self.weight_ho), torch.zeros_like(self.weight_ho)
        left_mask_for_W, right_mask_for_W = torch.zeros_like(self.W), torch.zeros_like(self.W)
        
        left_mask_for_weight_ih[hidden_size//2:, :] = 1 # everything on left is 1, used for left hemisphere activation
        right_mask_for_weight_ih[:hidden_size//2, :] = 1
        
        left_mask_for_weight_ho[:, hidden_size//2:] = 1 
        right_mask_for_weight_ho[:, :hidden_size//2] = 1
        
        left_mask_for_W[hidden_size//2:, :] = 1
        right_mask_for_W[:hidden_size//2, :] = 1
        
        self.mask = {'weight_ih':{'left':left_mask_for_weight_ih, 'right':right_mask_for_weight_ih, 'both':torch.ones_like(self.weight_ih)},
                     'weight_ho':{'left':left_mask_for_weight_ho, 'right':right_mask_for_weight_ho, 'both':torch.ones_like(self.weight_ho)},
                     'W':{'left':left_mask_for_W, 'right':right_mask_for_W, 'both':torch.ones_like(self.W)}}

        if hemisphere == None:
            self.hemisphere = 'both'
        else:
            self.hemisphere = hemisphere
            
        self.mode = mode
            
    def set_hemisphere(self, hemisphere):
        """
        Set the active hemisphere for training.
        'left' for left hemisphere, 'right' for right hemisphere, 'both' for both hemispheres.
        """
        assert hemisphere in ['left', 'right', 'both'], "Invalid hemisphere option"
        self.hemisphere = hemisphere
        print(f'Reset rnn hemisphere to {hemisphere}')
        
    def set_mode(self, mode):
        """
        Set the active hemisphere for training.
        'left' for left hemisphere, 'right' for right hemisphere, 'both' for both hemispheres.
        """
        assert mode in ['no-swap', 'swap'], "Invalid mode"
        self.mode = mode
        print(f'Reset mode to {mode}')  
        
    def forward(self, input, test_on, dis_on):

        # initialize state at the origin. randn is there just in case we want to play with this later.
        state = 0 * \
            torch.randn(input.shape[0], self.hidden_size, device=self.device)

        # defines process noise using Euler-discretization of stochastic differential equation defining the RNN
        noise = (
            1.41
            * self.process_noise
            * torch.randn(
                input.shape[0], input.shape[1], self.hidden_size, device=self.device
            )
        )

        # for storing RNN outputs and hidden states
        outputs = []
        states = []

        # loop over input
        for i in range(input.shape[1]):
            if self.mode == 'no-swap':
                if i >= test_on:
                    self.set_hemisphere('both')
                else:
                    self.set_hemisphere('right')
            elif self.mode == 'swap':
                if i < dis_on:
                    self.set_hemisphere('left')
                elif test_on > i >= dis_on:
                    self.set_hemisphere('right')
                else:
                    self.set_hemisphere('both')
            
            """masking for swap"""
            mask_for_weight_ih = self.mask['weight_ih'][self.hemisphere].to(self.weight_ih.device)
            mask_for_weight_ho = self.mask['weight_ho'][self.hemisphere].to(self.weight_ho.device)
            mask_for_W = self.mask['W'][self.hemisphere].to(self.W.device)
        
            # compute output
            hy = state @ (mask_for_weight_ho*self.weight_ho).T + self.bias_oh

            # save output and hidden states
            outputs += [hy]
            states += [state]

            # compute the RNN update
            fx = -state + self.phi(
                state @ ((mask_for_W*self.W) * self.struc_perturb_mask)
                + input[:, i, :] @ (mask_for_weight_ih*self.weight_ih).T
                + self.bias_hh
                + self.inv_sqrt_alpha * noise[:, i, :]
            )

            # step hidden state foward using Euler discretization
            state = state + self.alpha * (fx)

        # organize states and outputs and return
        return (
            torch.stack(outputs).permute(1, 0, 2),
            torch.stack(states).permute(1, 0, 2),
            noise,
            None,
        )


class aHiHRNNLayer(pl.LightningModule):
    """
    Network for anti-Hebbian / Inhibitory-Hebbian plasticity
    """

    def __init__(
        self, input_size, hidden_size, output_size, alpha, alpha_W, nonlinearity
    ):
        super(aHiHRNNLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.alpha = alpha
        self.alpha_W = alpha_W
        self.root_inv_alpha = 1 / np.sqrt(self.alpha)
        self.root_inv_hidden = 1 / np.sqrt(hidden_size)
        self.inv_hidden = 1 / hidden_size
        self.inv_hidden_power_4 = hidden_size ** (-0.25)

        self.root_inv_inp = 1 / np.sqrt(input_size)

        self.nonlinearity = nonlinearity
        if nonlinearity == "tanh":
            self.phi = torch.tanh
        if nonlinearity == "relu":
            self.phi = F.relu
        if nonlinearity == "none":
            self.phi = torch.nn.Identity()

        self.S = nn.Parameter(
            torch.FloatTensor(hidden_size, hidden_size).uniform_(-0.5, 0.5)
        )
        # self.register_buffer("K", torch.FloatTensor(hidden_size, hidden_size).uniform_(-.5, .5))
        self.gamma_val = 1/200
        self.register_buffer("gamma", self.gamma_val * torch.ones(1))
        self.register_buffer("beta", torch.ones(1))

        self.weight_ih = nn.Parameter(
            torch.FloatTensor(hidden_size, input_size).uniform_(
                -self.root_inv_inp, self.root_inv_inp
            )
        )

        self.weight_ho = nn.Parameter(
            torch.FloatTensor(output_size, hidden_size).uniform_(
                -self.root_inv_hidden, self.root_inv_hidden
            )
        )

        self.bias_oh = nn.Parameter(
            torch.FloatTensor(1, output_size).uniform_(
                -self.root_inv_hidden, self.root_inv_hidden
            )
        )
        self.bias_hh = nn.Parameter(
            torch.FloatTensor(1, hidden_size).uniform_(
                -self.root_inv_hidden, self.root_inv_hidden
            )
        )

        # self.half_I = 0.5*torch.eye(hidden_size,device = self.device)
        self.register_buffer("half_I", 0.5 * torch.eye(hidden_size))
        # self.half_I = self.half_I.type_as(self.half_I)

        # self.ones_mat = torch.ones((hidden_size,hidden_size),device = self.device)
        self.register_buffer("ones_mat", torch.ones(
            (hidden_size, hidden_size)))
        # self.ones_mat = self.ones_mat.type_as(self.ones_mat)

        self.eps = 0.95
        self.weight_inds_to_save_1 = torch.tensor(
            np.random.choice(np.arange(self.hidden_size), 50)
        ).long()
        self.weight_inds_to_save_2 = torch.tensor(
            np.random.choice(np.arange(self.hidden_size), 50)
        ).long()

        self.struc_p_0 = 0
        self.register_buffer(
            "struc_perturb_mask",
            torch.FloatTensor(self.hidden_size, self.hidden_size).uniform_()
            > self.struc_p_0,
        )
        self.process_noise = 0.05

    def forward(self, input):
        """Forward method for anti-Hebbian RNN, which is desribed by two coupled dynamical systems as desribed in Kozachkov et al (2020), PLoS Comp Bio:

        dxdt = -x + Wx + u(t)
        dWdt = -gamma W - K.*(xx')

        where K is positive semi-definite and has positive elements.

        """

        # initialize x,W at the origin
        x_state = torch.zeros(
            input.shape[0], self.hidden_size, device=self.device)

        W_state = torch.zeros(
            input.shape[0], self.hidden_size, self.hidden_size, device=self.device)

        # define neural process noise
        neural_noise = (
            1.41
            * self.process_noise
            * self.root_inv_alpha
            * torch.randn(
                input.shape[0], input.shape[1], self.hidden_size, device=self.device
            )
        )

        # for storing neural outputs, neural hidden states, and synaptic states

        outputs = []
        x_states = []
        W_states = []

        # loop over input dim
        for i in range(input.shape[1]):

            # compute and store neural output
            hy = x_state @ self.weight_ho.T + self.bias_oh
            outputs += [hy]

            # store neural hidden state
            x_states += [x_state]

            # store synaptic state
            W_states += [
                W_state[:, self.weight_inds_to_save_1,
                        self.weight_inds_to_save_2]
            ]

            #assert (W_state.transpose(1, 2) == W_state).all(), print("W is not symmetric!")

            # compute outer product in synaptic learning rule. use einsum magic to do it across batches.
            hebb_term = torch.einsum(
                "bq, bk-> bqk", self.phi(x_state), self.phi(x_state)
            )

            # compute K as the sum of three terms:
            # (S**2)'(S**2) ensures K is positive semidefinite and has non-negative elements
            # self.ones_mat ensures K has strictly positive elements
            # half_I ensures that K is positive-definite (not needed but it seems to help)
            K = (
                (((self.S) ** 2).T @ ((self.S) ** 2))
                + 1e-2 * self.ones_mat
                + 1e-2 * self.half_I
            )
            K *= self.struc_perturb_mask

            # batch matrix multiply W and x for updating x
            prod = torch.bmm(
                W_state, x_state.view(input.shape[0], self.hidden_size, 1)
            ).view(input.shape[0], self.hidden_size)

            # compute x update
            fx = -(self.beta) * x_state + self.phi(
                prod
                + input[:, i, :] @ self.weight_ih.T
                + self.bias_hh
                + neural_noise[:, i, :]
            )

            # compute W update
            fW = -K * hebb_term - (self.gamma) * W_state

            # step x and W forward
            x_state = x_state + self.alpha * (fx)
            W_state = W_state + self.alpha * (fW)

        # organize results and return

        return (
            torch.stack(outputs).permute(1, 0, 2),
            torch.stack(x_states).permute(1, 0, 2),
            torch.stack(W_states).permute(1, 0, 2),
            W_state,
        )

        '''
        return (
            torch.stack(outputs).permute(1, 0, 2),[],[],W_state
        )
        '''


class stspRNNLayer(pl.LightningModule):
    """Implements the RNN of Mongillo/Masse, using pre-synaptic STSP"""

    def __init__(
        self, input_size, hidden_size, output_size, alpha, dt, g, nonlinearity,
        hemisphere
    ):
        super(stspRNNLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.alpha = alpha
        self.inv_sqrt_alpha = 1 / np.sqrt(alpha)
        self.root_inv_hidden = 1 / np.sqrt(hidden_size)
        self.g = g
        self.dt = dt
        self.f_out = nn.Softplus()

        # define time-constants for the network, in units of ms
        self.tau_x_facil = 200
        self.tau_u_facil = 1500
        self.U_facil = 0.15

        self.tau_x_depress = 1500
        self.tau_u_depress = 200
        self.U_depress = 0.45

        # define nonlinearity for the neural dynamics
        self.nonlinearity = nonlinearity
        if nonlinearity == "tanh":
            self.phi = torch.tanh
        if nonlinearity == "relu":
            self.phi = F.relu
        if nonlinearity == "retanh":
            self.phi = torch.nn.ReLU(torch.nn.Tanh())
        if nonlinearity == "none":
            self.phi = torch.nn.Identity()

        # initialize input-to-hidden weights
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(hidden_size, input_size).uniform_(
                -self.root_inv_hidden, self.root_inv_hidden
            )
        )

        # initialize hidden to output weights
        self.weight_ho = nn.Parameter(
            torch.FloatTensor(output_size, hidden_size).uniform_(
                -self.root_inv_hidden, self.root_inv_hidden
            )
        )

        # initialize hidden-to-hidden weights
        W = torch.FloatTensor(hidden_size, hidden_size).uniform_(
            -self.root_inv_hidden, self.root_inv_hidden
        )
        W.log_normal_(0, self.g / np.sqrt(hidden_size))
        W = F.relu(W)
        W /= 10 * (torch.linalg.vector_norm(W, ord=2))
        self.W = nn.Parameter(W)
        
        """mask"""
        left_mask_for_weight_ih, right_mask_for_weight_ih = torch.zeros_like(self.weight_ih), torch.zeros_like(self.weight_ih)
        left_mask_for_weight_ho, right_mask_for_weight_ho = torch.zeros_like(self.weight_ho), torch.zeros_like(self.weight_ho)
        left_mask_for_W, right_mask_for_W = torch.zeros_like(self.W), torch.zeros_like(self.W)
        
        left_mask_for_weight_ih[hidden_size//2:, :] = 1 # everything on left is 1, used for left hemisphere activation
        right_mask_for_weight_ih[:hidden_size//2, :] = 1
        
        left_mask_for_weight_ho[:, hidden_size//2:] = 1 
        right_mask_for_weight_ho[:, :hidden_size//2] = 1
        
        left_mask_for_W[hidden_size//2:, :] = 1
        right_mask_for_W[:hidden_size//2, :] = 1
        
        self.mask = {'weight_ih':{'left':left_mask_for_weight_ih, 'right':right_mask_for_weight_ih, 'both':torch.ones_like(self.weight_ih)},
                     'weight_ho':{'left':left_mask_for_weight_ho, 'right':right_mask_for_weight_ho, 'both':torch.ones_like(self.weight_ho)},
                     'W':{'left':left_mask_for_W, 'right':right_mask_for_W, 'both':torch.ones_like(self.W)}}

        # define seperate inhibitory and excitatory neural populations
        diag_elements_of_D = torch.ones(self.hidden_size)
        diag_elements_of_D[int(0.8 * self.hidden_size):] = -1
        syn_inds = torch.arange(self.hidden_size)
        syn_inds_rand = torch.randperm(self.hidden_size)
        diag_elements_of_D = diag_elements_of_D[syn_inds_rand]
        D = diag_elements_of_D.diag_embed()

        self.register_buffer("D", D)

        self.register_buffer(
            "facil_syn_inds", syn_inds[: int(self.hidden_size / 2)])
        self.register_buffer("depress_syn_inds",
                             syn_inds[int(self.hidden_size / 2):])

        # time constants
        tau_x = torch.ones(self.hidden_size)
        tau_x[self.facil_syn_inds] = self.tau_x_facil
        tau_x[self.depress_syn_inds] = self.tau_x_depress
        self.register_buffer("Tau_x", (1 / tau_x))

        tau_u = torch.ones(self.hidden_size)
        tau_u[self.facil_syn_inds] = self.tau_u_facil
        tau_u[self.depress_syn_inds] = self.tau_u_depress
        self.register_buffer("Tau_u", (1 / tau_x))

        U = torch.ones(self.hidden_size)
        U[self.facil_syn_inds] = self.U_facil
        U[self.depress_syn_inds] = self.U_depress
        self.register_buffer("U", U)

        # initialize output bias
        self.bias_oh = nn.Parameter(
            0 * torch.normal(0, 1 / np.sqrt(hidden_size), (1, output_size))
        )

        # initialize hidden bias
        self.bias_hh = nn.Parameter(
            0 * torch.normal(0, 1 / np.sqrt(hidden_size), (1, hidden_size))
        )

        # for structurally perturbing weight matrix
        self.struc_p_0 = 0
        self.register_buffer(
            "struc_perturb_mask",
            torch.FloatTensor(self.hidden_size, self.hidden_size).uniform_()
            > self.struc_p_0,
        )
        self.process_noise = 0.05
        
        # New attribute to specify the active hemisphere
        self.hemisphere = hemisphere  # can be 'left', 'right', or 'both'

    def forward(self, input):
        # initialize neural state and synaptic states
        state = 0 * \
            torch.randn(input.shape[0], self.hidden_size, device=self.device)
        u_state = 0 * \
            torch.rand(input.shape[0], self.hidden_size, device=self.device)
        x_state = torch.ones(
            input.shape[0], self.hidden_size, device=self.device)

        # defines process noise
        noise = (
            1.41
            * self.process_noise
            * torch.randn(
                input.shape[0], input.shape[1], self.hidden_size, device=self.device
            )
        )
        
        """masking for swap"""
        # if self.hemisphere == 'left':
        mask_for_weight_ih = self.mask['weight_ih'][self.hemisphere].to(self.weight_ih.device)
        mask_for_weight_ho = self.mask['weight_ho'][self.hemisphere].to(self.weight_ho.device)
        mask_for_W = self.mask['W'][self.hemisphere].to(self.W.device)

        # for storing neural outputs, hidden states, and synaptic states
        outputs = []
        states = []
        states_x = []
        states_u = []

        for i in range(input.shape[1]):

            # compute and save neural output
            hy = state @ (mask_for_weight_ho*self.weight_ho).T + self.bias_oh
            outputs += [hy]

            # save neural and synaptic hidden states
            states += [state]
            states_x += [x_state]
            states_u += [u_state]

            # compute update for synaptic variables
            fx = (1 - x_state) * self.Tau_x - u_state * x_state * state * (
                self.dt / 1000
            )
            fu = (self.U - u_state) * self.Tau_u + self.U * (1 - u_state) * state * (
                self.dt / 1000
            )

            # define modulated presynaptic input based on STSP rule
            I = (x_state * state * u_state) @ (
                (self.D @ F.relu(mask_for_W*self.W)) * self.struc_perturb_mask
            )

            # compute neural update
            fstate = -state + self.phi(
                I
                + input[:, i, :] @ (mask_for_weight_ih*self.weight_ih).T
                + self.bias_hh
                + self.inv_sqrt_alpha * noise[:, i, :]
            )

            # step neural and synaptic states forward
            state = state + self.alpha * fstate
            x_state = torch.clamp(x_state + self.alpha * fx, min=0, max=1)
            u_state = torch.clamp(u_state + self.alpha * fu, min=0, max=1)

        # organize and return variables
        x_hidden = torch.stack(states_x).permute(1, 0, 2)
        u_hidden = torch.stack(states_u).permute(1, 0, 2)

        return (
            torch.stack(outputs).permute(1, 0, 2),
            torch.stack(states).permute(1, 0, 2),
            torch.cat((x_hidden, u_hidden), dim=2),
            noise,
        )
        
    # New method to set the active hemisphere
    def set_hemisphere(self, hemisphere):
        """
        Set the active hemisphere for training.
        'left' for left hemisphere, 'right' for right hemisphere, 'both' for both hemispheres.
        """
        assert hemisphere in ['left', 'right', 'both'], "Invalid hemisphere option"
        self.hemisphere = hemisphere
        print(f'Reset rnn hemisphere to {hemisphere}')
